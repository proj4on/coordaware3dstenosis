'''
This script is for testing given test data, calculate acc pre spe f1 and plot confusion matrix, roc and t-sne.
Including three parts, just uncite the part that you need.0
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import os
from sklearn.manifold import TSNE

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

from utils.evaluationmetrics import accuracy, roc, presenf1cfsmtx


model = torch.load('checkpoints/checkpoint1.pt')
device = torch.device('cuda:0')
model.to(device)
datapath = 'CoordConv_'


testdataset = Mydataset(split='fold1test', position='膝下段', path='CoordConv_')
testdataloader = DataLoader(testdataset, batch_size=2048, drop_last= False, shuffle = False)



# acc rec pre f1 confusion matrix
'''
acc = accuracy(model, testdataloader)
precision, recall, f1, truelist, predlist, cfsmtx = presenf1cfsmtx(model, testdataloader)

plt.title('CoordConv on above-knee arteries')
tick_marks = np.arange(4)
classes = ['0', '1', '2', '3',]
arange = 4
plt.imshow(cfsmtx, interpolation='nearest', cmap=plt.cm.Blues)
iters = np.reshape([[[i, j] for j in range(arange)] for i in range(arange)], (cfsmtx.size, 2))
for i, j in iters:
    plt.text(j, i, format(cfsmtx[i, j]), va='center', ha='center')

print('accuracy', acc, 'precision', precision, 'recall', recall, 'f1', f1)
plt.plot()
'''


# roc
'''
fpr, tpr, auc = roc(model, testdataloader)

plt.plot(fpr["micro"], tpr["micro"] ,label='coord aware noise robust net, AUC ' + str(round(100*auc["micro"],2)))
plt.title('ROC curves of different methods on above-knee arteries')
plt.legend()
plt.plot()
'''



#t-sne
'''
model.fc = nn.Sequential() 
img, label, _ = next(iter(testdataloader))
img = img.to(device)
a = model(img)

anp = a.detach().cpu().numpy()
labellist = label.tolist()

tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(anp)

colormap = ['red', 'green', 'blue', 'darkviolet', 'yellow', 'coral', 'dimgrey', 'pink', 'beige', 'lightsteelblue', 'maroon', 'olive']
colorlist = []
for i in labellist:
    colorlist.append(colormap[i])

plt.scatter(result[:, 0], result[:, 1], c=colorlist)
plt.xticks([])
plt.yticks([])
plt.title('Coord-Aware 3D Neural Network on Above-knee arteries')
'''


