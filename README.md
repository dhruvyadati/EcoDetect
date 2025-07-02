INTRODUCTION

This project uses an open source python package, DeepForest, to train and classify agricultural crops. 
DeepForest uses a deep learning object detection neural network and has pre-built model to predict bounding boxes of individual trees in RGB imagery. 
https://deepforest.readthedocs.io/en/v1.3.3/landing.html. 
The deepforest model is trained to perform object detection within bounding boxes for five crop classes: jute, maize, rice, sugarcane, wheat. 

The model also uses CropModel to train an object classification model to classify the detected boxes for the five crop types. 
The model was tested for variety of datasets including images from various perspectives.

The deepforest source code was installed locally and included in the base level of the project directory: https://github.com/weecology/DeepForest

INSTALLATION

Install deepforest from source code
https://deepforest.readthedocs.io/en/latest/installation.html

Install and activate conda packages
> conda deactivate 
> conda env remove -n env_name

> conda create -n deepforest python=3 pytorch torchvision -c pytorch
> conda install pytorch-lightning -c conda-forge

> conda activate deepforest
> conda install deepforest -c conda-forge
> conda install -c conda-forge albumentations
> conda install -c conda-forge supervision
> conda install huggingface_hub
> conda config --append channels conda-forge
> conda install numpy==1.22.4

Configure VSCode to use the correct Python:
In the terminal, find the current python version:
python --version

(deepforest) <dir>> % python --version
Python 3.12.2

In VSCode, press Ctrl+Shift+P (on Windows) then type:
Python: Select Interpreter
Use <CMD> + <Shift> + <P> on a Mac. 
Select the current python version

DATASETS

Images of the five crop types are downloaded from Kaggle datasets to "cropimages" directory. The trained images are written to the "data/crops_data" directory.

Source for kaggle: https://www.kaggle.com/datasets/aman2000jaiswal/agriculture-crop-images

Output of the predictions are saved to tiles_output csv files with the timestamp in the data directory.

CODE

Use TrainObjectDetectionAndClassifierAndPredictCrops.py to train the object detection and classification models, and predict test images against the models. 
The other files can be optionally used to pretrain, train, save and predict the models. 
PlotHistogramAccuracyScores.py gives a distribution of the predicted scores across the five classes. 
VisualizeBB.py can be used to visualize the predicted bounding boxes and classes on the test images.



