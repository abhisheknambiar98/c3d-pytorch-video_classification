import cv2
import math
import os
import re


import numpy as np
import torch
from os.path import join
from glob import glob
import skimage.io as io
# import matplotlib.pylab as plt
from skimage.transform import resize

import shutil

root="./data/UCF11_updated_mpg/"

def build_label_dict(label_src):
    label_dict={}
    c=0
    with open(str(label_src),"r") as f:
        for line in f:
            label_dict[line.rstrip()]=c
            c+=1
    return label_dict

cls_dict=build_label_dict('labels.txt')

def split_dataset():
    fl=open('./data/train.txt','w+')
    ft=open('./data/test.txt','w+')

    for f1 in os.listdir(root):
        cls_root=root+str(f1)+'/'
        f2=os.listdir(cls_root)
        f2.remove('Annotation')
        train=f2[:int(len(f2)*0.8)]
        test=f2[int(len(f2)*0.8):]
        for folder in train:
            for fname in os.listdir(cls_root+folder+'/'):
                shutil.copy2(cls_root+folder+'/'+fname,'./data/trainset/')
                fl.write(fname+' '+str(cls_dict[f1])+'\n')
        for folder in test:
            for fname in os.listdir(cls_root+folder):
                shutil.copy2(cls_root+folder+'/'+fname,'./data/testset/')
                ft.write(fname+' '+str(cls_dict[f1])+'\n')
    fl.close()
    ft.close()



train_root='./data/trainset/'


def captureFrame():
    folder='./data/trainframes/'
    file_dict={}
    tf=open('./data/train_frame.txt','w+')
    with open('./data/train.txt','r') as f:
        for line in f:
            li=line.split(' ')
            file_dict[li[0]]=int(li[1].rstrip())


    for name in os.listdir(train_root):
        cap = cv2.VideoCapture(train_root+name)
        frameRate = cap.get(5)
        x=1
        arr=[]
        name=name.split('.')[0]
        while(cap.isOpened()):
            frame_id = cap.get(0)
            ret, frame = cap.read()
            if (ret != True):
                break
            arr.append(frame)
            if x%16==0:
                frameset=folder+name+'_'+str(int(x/16))+'.jpg'
                tf.write(frameset+' '+str(file_dict[name+'.mpg'])+'\n')
                cv2.imwrite(frameset,get_clipframe(arr,True))
                arr=[]
            x+=1
        cap.release()
    tf.close()



def atoi(text):
    return int(text) if text.isdigit() else text

def sort_file(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_clipframe(clip,verbose=False):
    # clip = sorted(glob('./data/trainframes/'+clip_name +'*'),key=sort_file)   
    clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))   
        return clip_img
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    return clip


if __name__ == '__main__': 
	# captureFrame() 
    # split_dataset()
