import uuid
import argparse
import os
import cv2
import glob

def extractImages(pathIn, pathOut, num_ms=250):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*num_ms))    # added this line
        success, image = vidcap.read()
        if success:
            impath = os.path.join(pathOut, str(uuid.uuid4()) + ".jpg")
            cv2.imwrite(impath, image)     # save frame as JPEG file
        else:
            print(f"Read out {count} frames before end of file.")
        count += 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("pathIn", type=str, help="path to video")
    a.add_argument("pathOut", help="path to images")
    a.add_argument("--numbers", default=[131, 199, 271], type=int, nargs='+', help="numbers of milliseconds between images")
    a.add_argument("--video_suffix", default=".mp4", type=str, help="suffix to identify video files with glob")
    args = a.parse_args()
    video_files = glob.glob(args.pathIn + "*" + args.video_suffix)
    print(video_files)
    for file in video_files:
        for number in args.numbers:
            extractImages(file, args.pathOut, number)