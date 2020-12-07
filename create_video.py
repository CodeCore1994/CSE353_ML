import argparse
import os
import os.path
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision
import model
import dataloader
from tqdm import tqdm
import cv2 as cv

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help='path of video to be converted')
args = parser.parse_args()

def prepare_folders(path):
    if os.path.isdir(path):
        rmtree(path)
    os.mkdir(path)
    os.mkdir(os.path.join(path, "input"))
    os.mkdir(os.path.join(path, "output"))

def video_to_images(video, outDir):
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    videocap = cv.VideoCapture(video)
    success, image = videocap.read()
    count = 0

    while success:
        #code.interact(local=dict(globals(), **locals()))
        if "frame{}.png".format(count) in os.listdir(outDir):
            print("frame{}.png already in directory".format(count, count))
            success, image = videocap.read()
            count += 1
            continue

        cv.imwrite(os.path.join(outDir, "frame{:05d}.png".format(count)), image)
        print("{}: Saved frame {}".format(video, count))

        success, image = videocap.read()
        count += 1

def create_video(extractDir, outDir):
    img = cv.imread(os.path.join(extractDir, os.listdir(extractDir)[0]))
    height, width, layers = img.shape

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter('{}/output_video.mp4'.format(outDir), fourcc, 30, (width, height))

    for frame in sorted(os.listdir(extractDir)):
        img = cv.imread(os.path.join(extractDir, frame))
        print("writing frame {} into mp4".format(os.path.join(extractDir, frame)))
        video.write(img)

    cv.destroyAllWindows()
    video.release()

def main():
    extractPath = "./video_interpolation"
    prepare_folders(extractPath)

    video_to_images(args.video, os.path.join(extractPath, "input"))

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
    revNormalize = torchvision.transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1])

    if (device == "cpu"):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        TP = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
        TP = torchvision.transforms.Compose([revNormalize, torchvision.transforms.ToPILImage()])

    # Load data
    videoFrames = dataloader.Video(root=os.path.join(extractPath, "input"), transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=2, shuffle=False)

    # Initialize model
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load("./checkpoints/Interpolation0.ckpt", map_location=device)
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 0

    # batch_size = 2
    with torch.no_grad():
        for frameIndex, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            for batchIndex in range(2):
                (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(os.path.join(extractPath, "output"), "frame{:05d}.png".format(frameCounter + 2 * batchIndex)))
            frameCounter += 1

            # Generate intermediate frame

            t = float(1) / 2
            temp = -t * (1 - t)
            fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0

            g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

            wCoeff = [1 - t, t]

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            # Save intermediate frame
            for batchIndex in range(2):
                (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(os.path.join(extractPath, "output"), "frame{:05d}.png".format(frameCounter + 2 * batchIndex)))
            frameCounter += 1

            frameCounter += 2
    # Generate video from interpolated frames
    create_video(os.path.join(extractPath, "output"), os.path.join(extractPath, "output"))

if __name__ == "__main__":
    main()