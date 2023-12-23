import pandas as pd
from PIL import Image
import numpy as np
import os

import joblib
import torch
from torch.utils.data import Dataset

import h5py

import random
seed = 123456 #random seed for splitting training and testing set


import sklearn
from sklearn.model_selection import train_test_split


class Mydataset(Dataset):
    def __init__(self, transform=None, target_transform=None, split='train', position='all', path='traindata'):
        
        self.position = position
        self.split = split
        self.path = path
        
        if position == 'above' or position == 'all':
            self.num_classes=5
        if position == 'below':
            self.num_classes=4
        
        
        #Set the training and testing set for the thee fold cross valiation
        if 'fold' in split:
            originlist = os.listdir(self.path)
            self.shuffledlist = shuffledlist = originlist.copy()
            random.seed(seed)
            random.shuffle(shuffledlist)
            milestone1, milestone2 = int(len(shuffledlist)/3), int(len(shuffledlist)/3*2)  
            self.fold1testlist = fold1testlist = shuffledlist[0:milestone1]
            self.fold1trainlist = fold1trainlist = shuffledlist[milestone1:]
            
            self.fold2testlist = fold2testlist = shuffledlist[milestone1:milestone2]
            fold2trainlist = shuffledlist[0:milestone1]
            fold2trainlist.extend(shuffledlist[milestone2:])
            
            self.fold3testlist = fold3testlist = shuffledlist[milestone2:]
            self.fold3trainlist = fold3trainlist = shuffledlist[0:milestone2]
            
        if split == 'fold1train' or split == 'fold1val':
            self.filelist = fold1trainlist
        if split == 'fold1test':
            self.filelist = fold1testlist
        if split == 'fold2train' or split == 'fold2val':
            self.filelist = fold2trainlist
        if split == 'fold2test':
            self.filelist = fold2testlist
        if split == 'fold3train' or split == 'fold2val':
            self.filelist = fold3trainlist
        if split == 'fold3test':
            self.filelist = fold3testlist
        
        '''
        #Randomly split train and test set with sklearn. Not appliable in our experiments.
        if split == 'sktrain' or split == 'sktest':
            self.path = path
            alllist = os.listdir(path)
            trainlist, testlist = train_test_split(alllist, test_size = 1/3, random_state=seed)
        if split =='sktrain':
            self.filelist = trainlist
        if split == 'sktest':
            self.filelist = testlist
        '''
        
        '''
        #Manually split train and test set. Not appliable in our experiments.
        if split == 'train':
            self.path = path
            self.filelist = os.listdir(self.path)
        if split == 'test':
            self.path = path
            self.filelist = os.listdir(self.path)
        '''
        
        
        self.position = position

        self.imgtensorlist = []
        self.labellist = []
        self.filenamelist = []
        for i in range(len(self.filelist)):
            
            pkgfilename = self.filelist[i]
            pkgfilepath = os.path.join(self.path, pkgfilename)
            pkgdata = joblib.load(pkgfilepath).squeeze()
            pkgtensor = torch.from_numpy(pkgdata)

            if len(pkgtensor.shape) == 3: #for 3d resnet
                a,b,c = pkgtensor.shape
                pkgtensor = pkgtensor.reshape(1,a,b,c).float()
            if len(pkgtensor.shape) == 2: #for 2d resnet
                a,b = pkgtensor.shape
                pkgtensor = pkgtensor.reshape(1,a,b).float()
            
            '''
            #Above and below knee together, not applicable in our experiments.
            if position == 'all':
                self.imgtensorlist.append(pkgtensor)
                label = int(pkgfilename[-5:-4])
                self.labellist.append(label)
                self.filenamelist.append(pkgfilename)
            '''
            
            if position == 'above':
                if 'above' in pkgfilename:
                    self.imgtensorlist.append(pkgtensor)
                    label = int(pkgfilename[-5:-4])
                    self.labellist.append(label)
                    self.filenamelist.append(pkgfilename)
            
            
            if position == 'below':
                if 'below' in pkgfilename:
                    self.imgtensorlist.append(pkgtensor)
                    label = int(pkgfilename[-5:-4])
                    self.labellist.append(label)
                    self.filenamelist.append(pkgfilename)
                    

    def __getitem__(self, index):
        
        if 'val' in self.split:
            index = int(0.9*len(self.imgtensorlist)) + index

        return self.imgtensorlist[index], self.labellist[index], self.filenamelist[index]


    def __len__(self):
        
        if 'fold' in self.split and 'train' in self.split:
            length = int(0.9*len(self.imgtensorlist))
        elif 'fold' in self.split and 'val' in self.split:
            length = len(self.imgtensorlist) - int(0.9*len(self.imgtensorlist))
        else:
            length = len(self.imgtensorlist)
        return length