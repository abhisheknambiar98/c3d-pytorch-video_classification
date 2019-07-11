from __future__ import print_function

import os
import sys
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image


def one_hot(y,num_classes):
    y_hot=np.zeros(num_classes)
    y_hot[y]=1
    return torch.from_numpy(y_hot)


class UCF11(data.Dataset):
    def __init__(self, root, list_file, train, transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames=[]
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)
        
        for line in lines:
            splited=line.strip().split(' ')
            self.fnames.append(splited[0])
            self.labels.append(one_hot(int(splited[1]),num_classes=11))

    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) image index.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(fname)
        labels = self.labels[idx]


        img = self.transform(img)
        return img,labels


    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))   
    ])
    dataset = UCF11(root='./data/trainframes/', list_file='./data/train_frame.txt', train=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for images, labels in dataloader:
        images=torch.reshape(images,(1,3,112,16,112))
        images=torch.transpose(images,2,3)
        images=torch.transpose(images,1,2)
        print(images.shape)
        images=F.to_pil_image(images[0][0])
        # print(images.shape)
        images.show()
        print(labels)
        break

# test()    