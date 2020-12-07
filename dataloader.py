import torch.utils.data as data
from PIL import Image
import os
import os.path
import random

def _make_dataset(dir):

    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder) # output_skeleton/train/0
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))
    return framesPath

def _make_video_dataset(dir):

    framesPath = []
    # Find and loop over all the frames in root `dir`.
    for image in sorted(os.listdir(dir)):
        # Add path to list.
        framesPath.append(os.path.join(dir, image))
    return framesPath

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')
        #return flipped_img
    
    
class FIP(data.Dataset):

    def __init__(self, root, transform=None, dim=(640, 360), randomCropSize=(352, 352), train=True):

        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train

        self.framesPath     = framesPath

    def __getitem__(self, index):

        sample = []

        if (self.train):
            ### Data Augmentation ###
            # To select random 3 frames from 12 frames in a clip
            firstFrame = random.randint(0, 9)
            # Apply random crop on the input frames
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random reverse frame
            #frameRange = range(firstFrame, firstFrame + 9) if (random.randint(0, 1)) else range(firstFrame + 8, firstFrame - 1, -1)
            IFrameIndex = firstFrame + 1
            if (random.randint(0, 1)):
                frameRange = [firstFrame, IFrameIndex, firstFrame + 2]
                returnIndex = 0
            else:
                frameRange = [firstFrame + 2, IFrameIndex, firstFrame]
                returnIndex = 0
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSie[0], self.randomCropSize[1])
            IFrameIndex = 1
            returnIndex = IFrameIndex - 1
            frameRange = [0, IFrameIndex, 2]
            randomFrameFlip = 0
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
            # image = self.framesPath[index][frameIndex]
            # sample.append(image)
            
        return sample, returnIndex

    def __len__(self):

        return len(self.framesPath)

    def __repr__(self):

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class Video(data.Dataset):

    def __init__(self, root, transform=None):

        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_video_dataset(root)

        # Get dimensions of frames
        frame        = _pil_loader(framesPath[0])
        self.origDim = frame.size
        self.dim     = int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in: " + root + "\n"))

        self.root           = root
        self.framesPath     = framesPath
        self.transform      = transform

    def __getitem__(self, index):

        sample = []
        # Loop over for all frames corresponding to the `index`.
        for framePath in [self.framesPath[index], self.framesPath[index + 1]]:
            # Open image using pil.
            image = _pil_loader(framePath, resizeDim=self.dim)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample


    def __len__(self):

        # Using `-1` so that dataloader accesses only upto
        # frames [N-1, N] and not [N, N+1] which because frame
        # N+1 doesn't exist.
        return len(self.framesPath) - 1 

    def __repr__(self):

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str