# EcoDetect - Crop Detection and Classification

## INTRODUCTION

This project uses an open source python package, DeepForest, to train and classify agricultural crops. DeepForest uses a deep learning object detection neural network and has pre-built model to predict bounding boxes of individual trees in RGB imagery.

Documentation: https://deepforest.readthedocs.io/en/v1.3.3/landing.html

The deepforest model is trained to perform object detection within bounding boxes for five crop classes: jute, maize, rice, sugarcane, wheat.

The model also uses CropModel to train an object classification model to classify the detected boxes for the five crop types. The model was tested for variety of datasets including images from various perspectives.

The deepforest source code was installed locally and included in the base level of the project directory: https://github.com/weecology/DeepForest

## PROJECT STRUCTURE

```
EcoDetect/
├── README.md
├── TrainObjectDetectionAndClassifierAndPredictCrops.ipynb  # Training, detection, classification, and prediction
├── PredictionAnalysisForCanopyComplexity.ipynb             # Prediction analysis and canopy complexity study
├── GenerateBoundingBoxesWithFullSizes.py                   # Utility script for bounding box generation
└── data/
    ├── test_crop_image/                                    # 50 test images (10 per crop)
    ├── uav_crops_data_aerial/                              # Aerial dataset annotations
    └── uav_crops_data_multiperspective/                    # Multi-perspective dataset annotations and images
```

## INSTALLATION

### Option 1: Local Installation

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

### Option 2: Google Colab (Recommended)

The notebooks are designed to run in Google Colab with GPU support. Installation steps are included in the notebook cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhruvyadati/EcoDetect/blob/main/TrainObjectDetectionAndClassifierAndPredictCrops.ipynb)

**To run in Colab:**

1. Open the notebook in Colab using the badge above or by uploading the `.ipynb` file
2. Follow the numbered steps in the notebook comments (Steps 1-41)
3. The notebook handles all dependency installation automatically

## DATASET PREPARATION

### Image Source

Images of the five crop types are sourced from a public Kaggle dataset for agricultural classification.

Source: https://www.kaggle.com/datasets/aman2000jaiswal/agriculture-crop-images

### Colab Dataset Setup

Upload the following files to the Colab `/content` folder based on dataset type:

**For aerial dataset** (`multiperspective = False`):
- `crops_aerial_images.zip` - training images
- `crops_aerial_detection_annotations.csv` - detection bounding box annotations
- `crops_aerial_classification_annotations.csv` - classification annotations
- `test_crop_image.zip` - 50 test images for predictions

**For multi-perspective dataset** (`multiperspective = True`):
- `crops_multiperspective_images.zip` - training images
- `crops_multiperspective_detection_annotations.csv` - detection bounding box annotations
- `crops_multiperspective_classification_annotations.csv` - classification annotations
- `test_crop_image.zip` - 50 test images for predictions

**Important notes:**
- Detection training excludes full-frame bounding boxes (e.g., 0,0,224,224) since they provide no localization information
- Classification training uses full-frame images
- The `multiperspective` flag at the top of the notebook controls which dataset is used

## CODE USAGE

### Step 1: Train and Predict

Open `TrainObjectDetectionAndClassifierAndPredictCrops.ipynb` in Colab and run the cells sequentially. This notebook:

1. Installs dependencies and prepares the dataset
2. Trains a ResNet-50 classification model (CropModel) on cropped images
3. Trains a RetinaNet detection model (DeepForest) on bounding box annotations
4. Runs predictions on 50 held-out test images
5. Applies post-processing filters (size filter, sky filter) and generates annotated images
6. Saves trained models, prediction CSVs, and annotated images

Set `multiperspective = True` or `False` at the top of the notebook to select the dataset.

### Step 2: Prediction Analysis

Open `PredictionAnalysisForCanopyComplexity.ipynb` in Colab. Upload the prediction CSV files generated from Step 1 and run the cells. This notebook:

1. Adds true labels to prediction CSVs based on image filenames
2. Analyzes correct vs. incorrect predictions by score bucket (aerial and multi-perspective)
3. Computes per-class classification accuracy with confusion matrices
4. Performs canopy complexity analysis comparing multi-perspective vs. aerial accuracy
5. Runs statistical tests (Spearman rank correlation, one-way ANOVA)

### Utility Script

**GenerateBoundingBoxesWithFullSizes.py** - Utility to generate full-image-sized bounding boxes for classification annotations. Used to make loading compatible with the source CSV format.

## OUTPUT

- Trained model weights are saved as `.pt` files with timestamps
- Prediction results are saved as `tiles_output_*.csv` files
- Annotated images with bounding boxes are saved to `pred_bounding_boxed_images/`
- Merged prediction CSVs are generated for analysis
