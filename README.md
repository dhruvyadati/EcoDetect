# EcoDetect - Crop Detection and Classification

## INTRODUCTION

This project uses an open source python package, DeepForest, to train and classify agricultural crops. DeepForest uses a deep learning object detection neural network and has pre-built model to predict bounding boxes of individual trees in RGB imagery. 

Documentation: https://deepforest.readthedocs.io/en/v1.3.3/landing.html

The deepforest model is trained to perform object detection within bounding boxes for five crop classes: jute, maize, rice, sugarcane, wheat. 

The model also uses CropModel to train an object classification model to classify the detected boxes for the five crop types. The model was tested for variety of datasets including images from various perspectives.

The deepforest source code was installed locally and included in the base level of the project directory: https://github.com/weecology/DeepForest

## INSTALLATION

### Option 1: Local Mac Installation

Install deepforest from source code: https://deepforest.readthedocs.io/en/latest/installation.html

**Install and activate conda packages:**

```bash
conda deactivate 
conda env remove -n env_name

conda create -n deepforest python=3 pytorch torchvision -c pytorch
conda install pytorch-lightning -c conda-forge

conda activate deepforest
conda install deepforest -c conda-forge
conda install -c conda-forge albumentations
conda install -c conda-forge supervision
conda install huggingface_hub
conda config --append channels conda-forge
conda install numpy==1.22.4
```

**Configure VSCode to use the correct Python:**

In the terminal, find the current python version:
```bash
python --version
```

Example output:
```
(deepforest) <dir>> % python --version
Python 3.12.2
```

In VSCode, press Ctrl+Shift+P (on Windows) or CMD + Shift + P (on Mac), then type:
```
Python: Select Interpreter
```
Select the current python version.

### Option 2: Google Colab Installation

**Step 1: Restart Colab Runtime**

Go to "Runtime" → "Restart runtime". Do this before running any cells below.

**Step 2: Perform complete package uninstall**

```python
print("--- Starting a complete, aggressive uninstall of all related packages ---")
!pip uninstall -y \
pytorch-lightning \
torchmetrics \
pandas \
numpy \
scipy \
matplotlib \
scikit-learn \
deepforest \
deepforest-pytorch \
albumentations \
opencv-python \
Pillow \
fiona \
rasterio \
geopandas \
rtree \
shapely
```

**Step 3: Install deepforest from GitHub**

```python
print("\n--- Installing deepforest from GitHub main branch ---")
!pip install git+https://github.com/weecology/DeepForest.git@main
!pip install scikit-learn
```

**Step 4: Verify installations**

```python
print("\n--- Installed Versions ---")
!pip show numpy
!pip show pandas
!pip show torchmetrics
!pip show pytorch-lightning
!pip show albumentations
!pip show deepforest
```

## DATASET PREPARATION

### Local Dataset Setup

Images of the five crop types are downloaded from Kaggle datasets to "cropimages" directory. The trained images are written to the "data/crops_data" directory.

Source for kaggle: https://www.kaggle.com/datasets/aman2000jaiswal/agriculture-crop-images

### Colab Dataset Setup

**Required files to upload to /content folder:**

The following files need to be uploaded based on dataset type (multiperspective or aerial):

**For multiperspective dataset:**
- `crops_multiperspective_images.zip` - contains all training images
- `crops_multiperspective_detection_annotations.csv` - detection annotations for training images
- `crops_multiperspective_classification_annotations.csv` - classification annotations for training images
- `test_crop_image.zip` - contains all test images for predictions

**For aerial dataset:**
- `crops_aerial_images.zip` - contains all training images
- `crops_aerial_detection_annotations.csv` - detection annotations for training images
- `crops_aerial_classification_annotations.csv` - classification annotations for training images
- `test_crop_image.zip` - contains all test images for predictions

**Important notes:**
- Detection training excludes images where bounding box equals full image size (e.g., lines with 0,0,244,244)
- Full-size images can mislead RetinaNet as there is no detection training if bounding box covers the whole image
- Classification training uses full size images

**Dataset preparation code for Colab:**

```python
import os

# Set dataset type (True for multiperspective, False for aerial)
multiperspective = True  # Change this based on dataset

# Directory setup
if multiperspective:
    directory_path = "/content/data/uav_crops_data_multiperspective/"
    print("Configuring for Multiperspective dataset directory.")
else:
    directory_path = "/content/data/uav_crops_data_aerial/"
    print("Configuring for Aerial dataset directory.")

# Create directory
os.makedirs(directory_path, exist_ok=True)
print(f"Directory '{directory_path}' created successfully (or already exists).")

# Move annotation files
if multiperspective:
    detection_source_csv = "/content/crops_multiperspective_detection_annotations.csv"
    classification_source_csv = "/content/crops_multiperspective_classification_annotations.csv"
    zip_file = "crops_multiperspective_images.zip"
else:
    detection_source_csv = "/content/crops_aerial_detection_annotations.csv"
    classification_source_csv = "/content/crops_aerial_classification_annotations.csv"
    zip_file = "crops_aerial_images.zip"

# Move annotation files
!mv {detection_source_csv} {directory_path}
!mv {classification_source_csv} {directory_path}

# Unzip training images
!unzip {zip_file} -d {directory_path}

# Unzip test images
!unzip test_crop_image.zip -d /content/data/
```

## CODE USAGE

Follow installation and dataset preparation steps in README, then execute the following steps:

### Step 1: Train and Predict
```bash
python TrainObjectDetectionAndClassifierAndPredictCrops.py
```
Main class that trains the 1) detection, 2) classification models and predicts test crop images against the trained models. The prediction can be tested with different datasets:
- `multiperspective = True` expects images taken with multiple perspectives
- `multiperspective = False` expects images taken only with aerial imagery

### Step 2: Visualize Results
```bash
python VisualizeBoundingBoxesOfPredictedImages.py
```
Run this to visualize previously predicted and archived image files. Unzip the previously loaded images into pred_output directory.

### Step 3: Generate Analysis CSV 
```bash
python CreatePredictedOutputScoresCSV.py
```
Create a final CSV file with all the scores by merging all the individual tiles output csv files. This CSV file will be used to run analysis on accuracy scores of predictions.

### Step 4: Export Results 
```bash
python CreatePredictedOutputZip.py
```
Zip the predicted output files for download. Also remember to download the trained models.

### Step 5: Prediction Analysis 
```bash
python PredictionAnalysisCorrectVsIncorrect.py
python PredictionAnalysisForClassificationAccuracy.py
python PredictionAnalysisForCanopyComplexity.py
```
Upload the predicted csv output files to Colab for analysis.


### Optional Utility Files

**GenerateBoundingBoxesWithFullSizes.py**
```bash
python GenerateBoundingBoxesWithFullSizes.py
```
Utility file to generate full-image-sized bounding boxes to make loading compatible with the source csv file format. This is for loading the full-sized images to train classification model. Full-sized images are not used in the detection model.

**PlotHistogramForDatasetSizes.py**
```bash
python PlotHistogramForDatasetSizes.py
```
Utility file to display dataset distribution.

## OUTPUT

Output of the predictions are saved to tiles_output CSV files with timestamps in the data directory.
