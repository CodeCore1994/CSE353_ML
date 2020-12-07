import os
import os.path
import sys
from shutil import rmtree, move
import random
import code
import cv2 as cv
import numpy as np
import json

dir_path = os.path.dirname(os.path.realpath(__file__))

#OpenPose Configuration
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

def prepare_folders():
    if os.path.isdir('./output_skeleton_60fps'):
        rmtree("./output_skeleton_60fps")
    os.mkdir('./output_skeleton_60fps')

    os.mkdir("./output_skeleton_60fps/extracted")
    os.mkdir("./output_skeleton_60fps/train")
    os.mkdir("./output_skeleton_60fps/train_keyPoints")
    os.mkdir("./output_skeleton_60fps/test")
    os.mkdir("./output_skeleton_60fps/test_keyPoints")
    os.mkdir("./output_skeleton_60fps/validation")
    os.mkdir("./output_skeleton_60fps/validation_keyPoints")

def create_clips(root, destination):
    folderCounter = -1
    files = os.listdir(root)
    for file in files:
        images_and_json = sorted(os.listdir(os.path.join(root, file)))
        # code.interact(local=dict(globals(), **locals()))
        for imageCounter, image in enumerate(images_and_json):
            # Bunch images in groups of 12 frames --> 24 files including corresponding json files
            if (imageCounter % 24 == 0):
                if (imageCounter + 23 >= len(images_and_json)):
                    break
                folderCounter += 1
                os.mkdir("{}/{}".format(destination, folderCounter))
                os.mkdir("{}_keyPoints/{}".format(destination, folderCounter))
            if (imageCounter % 2 == 0):
                move("{}/{}/{}".format(root, file, image), "{}/{}/{}".format(destination, folderCounter, image))
            else:
                move("{}/{}/{}".format(root, file, image), "{}_keyPoints/{}/{}".format(destination, folderCounter, image))
        rmtree(os.path.join(root, file))

def video_to_images(videos, inDir, outDir):
    for video in videos:
        new_directory = os.path.join(outDir, os.path.splitext(video)[0])
        if not os.path.isdir(new_directory):
            os.mkdir(new_directory)
        videocap = cv.VideoCapture(os.path.join(inDir, video))
        success, image = videocap.read()
        count = 0

        while success:
            #code.interact(local=dict(globals(), **locals()))
            if "frame{}.jpg".format(count) in os.listdir(new_directory) and "frame{}.json".format(count) in os.listdir(new_directory):
                print("frame{}.jpg and frame{}.json already in directory".format(count, count))
                success, image = videocap.read()
                count += 1
                continue

            cv.imwrite(os.path.join(new_directory, "frame{:05d}.jpg".format(count)), image)
            print("{}: Saved frame {}".format(video, count))

            datum = op.Datum()
            datum.cvInputData = image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            json_file = os.path.join(new_directory, "frame{:05d}.json".format(count))
            with open(json_file, "w") as f:
                try:
                    json.dump(datum.poseKeypoints.tolist(), f)
                except:
                    pass

            success, image = videocap.read()
            count += 1

def create_test_dataset():
    f = open("data/60fps/test_list.txt", "r")
    videos = f.read().split('\n')
    video_to_images(videos, "./data/60fps/original_60_fps_videos", "./output_skeleton_60fps/extracted")
    create_clips("./output_skeleton_60fps/extracted", "./output_skeleton_60fps/test")
    f.close()

def create_train_dataset():
    f = open("data/60fps/train_list.txt", "r")
    videos = f.read().split('\n')
    video_to_images(videos, "./data/60fps/original_60_fps_videos", "./output_skeleton_60fps/extracted")
    create_clips("./output_skeleton_60fps/extracted", "./output_skeleton_60fps/train")
    f.close()

def create_validate_dataset():  ###
    testClips = os.listdir("./output_skeleton_60fps/test")
    indices = random.sample(range(len(testClips)), 100)
    for index in indices:
        move("{}/{}".format("./output_skeleton_60fps/test", index), "{}/{}".format("./output_skeleton_60fps/validation", index))
        move("{}/{}".format("./output_skeleton_60fps/test_keyPoints", index), "{}/{}".format("./output_skeleton_60fps/validation_keyPoints", index))

if __name__ == "__main__":
    # code.interact(local=dict(globals(), **locals()))
    prepare_folders()
    create_test_dataset()
    create_train_dataset()
    create_validate_dataset()
