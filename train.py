from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from model import C3D
from dataset import UCF11


parser = argparse.ArgumentParser(description='PyTorch C3D Training')
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])


batch_size = 16
validation_split = 0.2
shuffle_dataset = True
random_seed= 42


trainset = UCF11(root='./data/trainframes/',list_file='./data/train_frame.txt', train=True, transform=transform)

#creating train and train-val sets
dataset_size = len(trainset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler=train_sampler,num_workers=8)
validation_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler=valid_sampler,num_workers=8)


#creating network

net=C3D()

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()


criterion=nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)



def train(epoch):
    # for epoch in range(start_epoch+50):
        print("\nTraining Epoch: ",epoch)
        train_loss = 0.0
        for batch_id, (inputs,labels) in enumerate(train_loader):
            inputs=torch.reshape(inputs,(batch_size,3,112,16,112))
            inputs=torch.transpose(inputs,2,3)

            inputs=inputs.cuda()
            labels=labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            print('train_loss: %.3f | avg_loss: %.3f' % (loss.item(), train_loss/(batch_id+1)))

def test(epoch):
    print('\nValidation!')
    net.eval()
    val_loss = 0
    for batch_id, (inputs,labels) in enumerate(validation_loader):
        inputs=torch.reshape(inputs,(batch_size,3,112,16,112))
        inputs=torch.transpose(inputs,2,3)

        with torch.no_grad():

            inputs = inputs.cuda()
            labels=labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            print('val_loss: %.3f | avg_loss: %.3f' % (loss.item(), val_loss/(batch_id+1)))

    # Save checkpoint
    global best_loss
    val_loss /= len(validation_loader)
    if val_loss < best_loss:
        print('Saving checkpoint..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = val_loss


for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    test(epoch)


# train(1)