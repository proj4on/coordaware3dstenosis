This is the code implementation of 'Coordinate aware three dimensional neural network for stenosis classification of lower limb arteries'. 

How to use:

Step 0. Set up running environment:
Install the packages listed in requirements.txt

Step 1. Crop patches from CTA
Run crop_patches_from_CTA.py in python. This script will generate and save 2d, 3d and 3d coordinat patches with given CTA and ROI coordinate.

Step 2. Train
Run main.py in python, this will train the neural network model with given training data in a three-fold cross validation, save the checkpoint and output the accuracy score on the test test.

Step 3. Test
Run test_one_sample.py, this will output the predicted stenosis level with given neural network checkpoint and given patch.
Or: Run evaluate_and_plot.py or ipynb. This python script will calculate the prediction output with a saved checkpoint and give test data (more than one sample), calculate the accuracy score, sensitivity, specificity, F1, and plot confusion matrix, ROC curve and t-SNE plot.


Some sample patches are provided in sample_patches folder. 
Pretrianed checkpoints can be downloaded at: https://drive.google.com/drive/folders/1HcJ0eyATYgC7H1C40-ooA3bHDUWGscgO?usp=sharing



