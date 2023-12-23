'''
#This script for testing the classification result of the neural network with one sample.
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
from nets.yixianresnet import resnet18 as threed_resnet18
from nets.blingblingresnet import resnet18 as twod_resnet18



checkpointpath = '*your_checkpoint_path*'   #Path of the saved checkpoint
patchpath = '*your_patch_path*'   #Path of the patch


device = torch.device('cpu')

model = torch.load(checkpointpath).to(device)
data = joblib.load(patchpath)

datatensor = torch.from_numpy(data)
if len(datatensor.shape) ==2: #for 2d data
    a,b = datatensor.shape
    datatensor = datatensor.reshape(1,1,a,b).float()
if len(datatensor.shape) ==3: #for 3d data
    a,b,c = datatensor.shape
    datatensor = datatensor.reshape(1,1,a,b,c).float()

model.eval
output = model(datatensor)
prediction = torch.max(output, 1)[1].to(device)

print('Prediction is: type', prediction.item())

