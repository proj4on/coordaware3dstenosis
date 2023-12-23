'''
#This python script is for cropping patches from raw CTA image for training and testing
#This script will generate testpatch.pkg in the root file.

preliminary：
suppose x1, x2, y1, y2, z1, z2 are the coordinates of boundaries of the ROI in the CTA image.
path is the folder that contains the CTA image files that end with .DCM

this python script will crop coordconv pathces, 3d patches and 2d patches and save them in the current folder.
'''

import SimpleITK as sitk
import numpy as np
import json
import os
import joblib



#path of dicom data
path = '王鸣江'
#coordinates of boundaries of one ROI, sized 16*16*16. Like:
x1 = 50
x2 = 66
y1 = 50
y2 = 66
z1 = 50
z2 = 66


#a function to read dicom files with sitk
def readdicom(path):  
    reader = sitk.ImageSeriesReader()
    case_path = path  
    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute() 
    #image_array = sitk.GetArrayFromImage(image) # z, y, x   
    size = image.GetSize() 
    origin = image.GetOrigin() 
    spacing = image.GetSpacing()
    direction = image.GetDirection() 
    correctmatrix = np.array([
        [-1.0000, 0.0000, 0.0000],
        [0.0000, -1.0000, 0.0000],
        [0.0000, 0.0000, 1.0000]])
    correctorigin = np.array(origin)
    correctsize = np.array(size)
    correctspace = np.array(spacing)
    return correctmatrix, correctorigin, correctsize, correctspace, image



matrix, origin, size, space, sitkdata = readdicom(path)  #Read dicom

sitkarray = sitk.GetArrayFromImage(sitkdata)
sitkswappedarray = np.swapaxes(sitkarray, 0, 2)


sitkslice = sitkswappedarray[x1:x2, y1:y2, z1:z2]
sitkslice = sitkslice.astype('int16')     #3D patch


xcoord = np.empty([16, 16, 16])  
ycoord = np.empty([16, 16, 16])
zcoord = np.empty([16, 16, 16])
xmax = size[0]
ymax = size[1]
zmax = size[2]

for xi in range(xcoord.shape[0]):
    xtemp = x1+xi
    xpercent = xtemp/xmax
    slicetemp = np.ones([16,16]) * xpercent
    xcoord[xi,:,:] = slicetemp
for yi in range(ycoord.shape[0]):
    ytemp = y1+yi
    ypercent = ytemp/ymax
    slicetemp = np.ones([16,16]) * ypercent
    ycoord[:,yi,:] = slicetemp        
for zi in range(zcoord.shape[0]):
    ztemp = z1+zi
    zpercent = ztemp/zmax
    slicetemp = np.ones([16,16]) * zpercent
    zcoord[:,:,zi] = slicetemp             
finalpkgslice = np.empty([16,16,64])     #Coord Conv patch
for i in range(sitkslice.shape[2]):
    finalpkgslice[:,:,4*i] = sitkslice[:,:,i] /500
    finalpkgslice[:,:,4*i+1] = xcoord[:,:,i]
    finalpkgslice[:,:,4*i+2] = ycoord[:,:,i]
    finalpkgslice[:,:,4*i+3] = zcoord[:,:,i]


d2slice = sitkswappedarray[x1:x2, y1:y2, z1:z1+1]
d2slice = d2slice.astype('int16')      #2D patch

#save the above three generated patches
joblib.dump(d2slice, 'sample 2d patch.pkg', compress=0)
joblib.dump(sitkslice, 'sample 3d patch.pkg', compress=0)
joblib.dump(finalpkgslice, 'sample coordconv 3d patch.pkg', compress=0)


#Put the generated pathces in 'traindata' folder for training.







