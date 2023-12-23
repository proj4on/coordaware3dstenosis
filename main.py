'''
This script is for training and testing the neural network with cropped patches and saving the checkpoints.
Make sure the file name of training patches end with "*type*.pkg", like: case_8096_type_1.pkg, where 1 indicates stenosis level.
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn.functional as F

from utils.dataload import Mydataset
from nets.threed_resnet import resnet18 as threed_resnet18
from nets.twod_resnet import resnet18 as twod_resnet18



#Use this to set the position (above or below knee) and fold of the three-fold experiment, and the path that contain the training data
traindataset = Mydataset(split='fold1train', position='above', path='*your_data_path*')
testdataset = Mydataset(split='fold1test', position='above', path='*your_data_path*')

a,b,c = traindataset[0] #For 3D resnet (our CoordConv model and 3d basline model)
if len(a.shape) == 4:
    model = threed_resnet18(num_classes = traindataset.num_classes)
if len(a.shape) == 3: #For 2D resnet, which is only for comparison
    model = twod_resnet18(num_classes = traindataset.num_classes)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

error = nn.CrossEntropyLoss()
batchsize=1024
learning_rate = 0.002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 250


traindataloader = DataLoader(traindataset, batch_size=batchsize, drop_last= False)
testdataloader = DataLoader(testdataset, batch_size=batchsize, drop_last= False, shuffle = False)


#train
for epoch in range(num_epochs):
    print('training epoch', epoch+1)
    model.train()
    for images, labels, _ in traindataloader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()
        outputs = model(images)
        loss = error(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#test
model.eval()
total = 0
correct = 0
for images, labels, _ in testdataloader:
    images, labels = images.to(device), labels.to(device)
    labels = labels.long()
    outputs = model(images)
    predictions = torch.max(outputs, 1)[1].to(device)
    correct += (predictions == labels).sum()
    total += len(labels)
acc = correct.item() / total
print('accuracy', acc)


#torch.save(model, 'checkpoints/checkpoint.pt')
