import cv2
import math
import numpy as np
from os.path import join
from glob import glob
import skimage.io as io
from skimage.transform import resize


def FrameCapture(path):  
    cap = cv2.VideoCapture(path)
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate/4) == 0):
            filename = './frames/frame' +  str(int(x)) + ".jpg";x+=1
            cv2.imwrite(filename, frame)

    cap.release()

def get_clipframe(clip_name,verbose=True):
    clip = sorted(glob(join('data', clip_name, '*.jpg')))
    clip = np.array([resize(cv2.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    print(clip.shape)
     

# Driver Code 
if __name__ == '__main__': 
	FrameCapture("./data/ucf11/basketball/v_shooting_01/v_shooting_01_01.avi") 
