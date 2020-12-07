import torch
import torchvision
import model
import dataloader
import datetime
import time, code
from tensorboardX import SummaryWriter
import os
import math
import sys
import cv2 as cv
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

### OpenPose Configuration
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/bin/python/openpose/Release');
    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' + dir_path + '/bin;'
    import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "./models/"

# Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

writer = SummaryWriter('test_log')

###Loss and Optimizer
L1_lossFn = torch.nn.L1Loss()
MSE_LossFn = torch.nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn1 = model.UNet(6, 4)
cnn2 = model.UNet(20, 5)
cnn1.to(device)
cnn2.to(device)

###Initialze backward warpers for train and validation datasets
trainFlowBackWarp = model.backWarp(352, 352, device)
trainFlowBackWarp = trainFlowBackWarp.to(device)
validationFlowBackWarp = model.backWarp(640, 352, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)

###Create transform to display image from tensor
revNormalize = torchvision.transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1])
TP = torchvision.transforms.Compose([revNormalize, torchvision.transforms.ToPILImage()])

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv = torch.nn.Sequential(*list(vgg16.children())[0][:29])
vgg16_conv.to(device)

# Freeze convolutional weights
for param in vgg16_conv.parameters():
    param.requires_grad = False

###Utils
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def loadData():
    normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    trainset = dataloader.FIP(root='./output_skeleton_60fps/train', dim=(1280, 720), transform=transform, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)

    validationset = dataloader.FIP(root='./output_skeleton_60fps/validation', dim=(1280, 720), transform=transform, randomCropSize=(640, 352),
                                   train=False)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=2, shuffle=False)

    return trainloader, validationloader


def validate(validationloader):
    psnr = 0
    tloss = 0

    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)

            flowOut = cnn1(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = model.getFlowCoeff(validationFrameIndex, device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)

            intrpOut = cnn2(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = torch.nn.functional.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)

            wCoeff = model.getWarpCoeff(validationFrameIndex, device)

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            # For tensorboard
            if (validationIndex == 20):
                retImg1 = torchvision.utils.make_grid(
                    [revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0])], padding=10)
            if (validationIndex == 40):
                retImg2 = torchvision.utils.make_grid(
                    [revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0])], padding=10)

            # loss
            recnLoss = L1_lossFn(Ft_p, IFrame)

            prcpLoss = MSE_LossFn(vgg16_conv(Ft_p), vgg16_conv(IFrame))

            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(
                validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)

            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
                torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
                torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            print("validationIndex: {}, loss: {}".format(validationIndex, loss))

            tloss += loss.item()

            # psnr
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * math.log10(1 / MSE_val.item()))

    return (psnr / len(validationloader)), (tloss / len(validationloader)), retImg1, retImg2


def train(trainloader, validationloader):
    ### Initialization
    params = list(cnn2.parameters()) + list(cnn1.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    start = time.time()
    cLoss = dict1['loss']
    valLoss = dict1['valLoss']
    valPSNR = dict1['valPSNR']

    checkpoint_counter = 0

    for epoch in range(dict1['epoch'] + 1, 3):
        print("Epoch: ", epoch)

        # Append and reset
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iLoss = 0

        # Increment scheduler count
        scheduler.step()

        for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):

            ## Getting the input and the target from the training set
            frame0, frameT, frame1 = trainData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)

            optimizer.zero_grad()

            # Calculate flow between reference frames I0 and I1
            flowOut = cnn1(torch.cat((I0, I1), dim=1))

            # Extracting flows between I0 and I1 - F_0_1 and F_1_0
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = model.getFlowCoeff(trainFrameIndex, device)

            # Calculate intermediate flows
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)

            # Calculate optical flow residuals and visibility maps
            intrpOut = cnn2(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            # Extract optical flow residuals and visibility maps
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = torch.nn.functional.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)

            wCoeff = model.getWarpCoeff(trainFrameIndex, device)

            # Calculate final intermediate frame
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            interpolated_keypoints = []
            intermediate_keypoints = []

            for batchNum in range(2):
                try:
                    TP(IFrame[batchNum].cpu().detach()).save("intermediate{}.jpg".format(batchNum))
                    datum = op.Datum()
                    image = cv.imread("./intermediate{}.jpg".format(batchNum))
                    datum.cvInputData = image
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    if datum.poseKeypoints is None:
                        keypoints = np.zeros([25, 3], dtype=float)
                    else:
                        keypoints = datum.poseKeypoints[0]

                    intermediate_keypoints.append([keypoints])
                except:
                    pass

            for batchNum in range(2):
                try:
                    TP(Ft_p[batchNum].cpu().detach()).save("interpolated{}.jpg".format(batchNum))
                    datum = op.Datum()
                    image = cv.imread("./interpolated{}.jpg".format(batchNum))
                    datum.cvInputData = image
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    if datum.poseKeypoints is None:
                        keypoints = np.zeros([25, 3], dtype=float)
                    else:
                        keypoints = datum.poseKeypoints[0]

                    interpolated_keypoints.append([keypoints])
                except:
                    pass

            intermediate_keypoints = torch.tensor(intermediate_keypoints).float().to(device)
            interpolated_keypoints = torch.tensor(interpolated_keypoints).float().to(device)

            # Loss
            recnLoss = L1_lossFn(Ft_p, IFrame)

            prcpLoss = MSE_LossFn(vgg16_conv(Ft_p), vgg16_conv(IFrame))

            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(
                trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)

            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
                torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
                torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

            keypointsLoss = MSE_LossFn(interpolated_keypoints, intermediate_keypoints)
            # code.interact(local=dict(globals(), **locals()))
            # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
            # since the loss in paper is calculated for input pixels in range 0-255
            # and the input to our network is in range 0-1
            loss = 163 * recnLoss + 82 * warpLoss + 0.004 * prcpLoss + 0.8 * loss_smooth + 0.001 * keypointsLoss
            print("trainIndex: {}, loss: {}".format(trainIndex, loss))

            # Backpropagate
            loss.backward()
            optimizer.step()
            iLoss += loss.item()

            # Validation and progress every 50 iterations
            if ((trainIndex % 50) == 49):
                end = time.time()

                psnr, vLoss, valImg1, valImg2 = validate(validationloader)

                valPSNR[epoch].append(psnr)
                valLoss[epoch].append(vLoss)

                # Tensorboard
                itr = trainIndex + epoch * (len(trainloader))

                writer.add_scalars('Loss', {'trainLoss': iLoss / 50,
                                            'validationLoss': vLoss}, itr)
                writer.add_scalar('PSNR', psnr, itr)

                writer.add_image('Validation1', valImg1, itr)
                writer.add_image('Validation2', valImg2, itr)

                #####

                endVal = time.time()

                print(
                    " Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (
                    iLoss / 50, trainIndex, len(trainloader), end - start, vLoss, psnr, endVal - end,
                    get_lr(optimizer)))

                cLoss[epoch].append(iLoss / 50)
                iLoss = 0
                start = time.time()

        # Create checkpoint after every 3 epochs
        if (not os.path.isdir('./checkpoints')):
            os.mkdir('./checkpoints')
        if ((epoch % 3) == 2):
            dict1 = {
                'Detail': "End to end Super SloMo.",
                'epoch': epoch,
                'timestamp': datetime.datetime.now(),
                'trainBatchSz': 2,
                'validationBatchSz': 2,
                'learningRate': get_lr(optimizer),
                'loss': cLoss,
                'valLoss': valLoss,
                'valPSNR': valPSNR,
                'state_dictFC': cnn1.state_dict(),
                'state_dictAT': cnn2.state_dict(),
            }
            torch.save(dict1, "./checkpoints/Interpolation" + str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1


if __name__ == "__main__":
    trainloader, validationloader = loadData()
    train(trainloader, validationloader)