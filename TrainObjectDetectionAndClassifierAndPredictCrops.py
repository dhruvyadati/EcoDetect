# Main class that trains the 1) detection, 2) classification models
# and predicts test crop images against the trained models.
# The prediction can be tested with different datasets:
# multiperspective = True expects images taken with multiple perspectives. 
# multiperspective = False expects images taken only with aerial imagery. 

#load the modules
import os
import time
import torch
import pandas as pd
import datetime
import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt
from deepforest import main
from deepforest import get_data
from deepforest import utilities
from deepforest import preprocess
from deepforest import model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# --- Parameter Definition ---
multiperspective = True # Set to True for multiperspective, False for aerial
# --- End of Parameter Definition ---

# ---------------------------------------
# Write cropped images to traning folder based on training images and bounding box annotations

# --- Path Definitions based on parameter ---
if multiperspective:
    cwd = "/content/data/uav_crops_data_multiperspective"
    # Updated to use the classification annotations CSV
    csv_filename = "crops_multiperspective_classification_annotations.csv"
    root_images_folder = "crops_multiperspective_images"
    trained_crops_folder = "trained_multiperspective_crops"
    print("Configuring paths for Multiperspective classification data.")
else:
    cwd = "/content/data/uav_crops_data_aerial"
    # Updated to use the classification annotations CSV (assuming similar naming)
    csv_filename = "crops_aerial_classification_annotations.csv"
    root_images_folder = "crops_aerial_images"
    trained_crops_folder = "trained_aerial_crops"
    print("Configuring paths for Aerial classification data.")

csv_file_path = os.path.join(cwd, csv_filename)
root_dir = os.path.join(cwd, root_images_folder)
crops_dir = os.path.join(cwd, trained_crops_folder)
os.makedirs(crops_dir, exist_ok=True)
# --- End of Path Definitions ---

print(f"Using multiperspective dataset: {multiperspective}")
print(f"Base directory (cwd): {cwd}")
print(f"CSV file path: {csv_file_path}")
print(f"Root directory for original images: {root_dir}")
print(f"Directory for saving cropped images: {crops_dir}")

# Setting up training ResNet-50 Classification model for cropped images
# num_classes matches dataset's actual number of classes
crop_model = model.CropModel(num_classes=5)

# Read the CSV
df = pd.read_csv(csv_file_path)

# --- IMPORTANT: Ensure bounding box columns are numeric ---
bbox_columns = ['xmin', 'ymin', 'xmax', 'ymax']
for col in bbox_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=bbox_columns, inplace=True)
df[bbox_columns] = df[bbox_columns].astype(int)
# --- End of numeric conversion ---

try:
    boxes = df[bbox_columns].values.tolist()
except Exception as e:
    print(f"Error extracting boxes after conversion: {e}")
    boxes = []

images = df.image_path.values

print("Calling write_crops")
crop_model.write_crops(boxes=boxes,
                        labels=df.label.values, # Assuming 'df.label' contains the class labels for each box
                        root_dir=root_dir,
                        images=images,
                        savedir=crops_dir)

print("Crop writing process complete.")

# ---------------------------------------
# Create a trainer
# fast_dev_run=true flips from deep training to one train.
# False will run the trainer for multiple batches and epochs
crop_model.create_trainer(fast_dev_run=False, max_epochs=10)
print("Creating trainer is complete.")



print("crops_dir:")
print(crops_dir)
crop_model.load_from_disk(train_dir=crops_dir, val_dir=crops_dir)

# Test training dataloader
train_loader = crop_model.train_dataloader()
assert isinstance(train_loader, torch.utils.data.DataLoader)

# Test validation dataloader
val_loader = crop_model.val_dataloader()
assert isinstance(val_loader, torch.utils.data.DataLoader)

print("Setting up train loader and val loader is complete.")


# Training and Evaluation using GPU
# TRAIN & VALIDATE RestNet-50 classification model; 85/15 train/test split
crop_model.trainer.fit(crop_model)
crop_model.trainer.validate(crop_model)

print("Training and validating crop_model done.")


# ---------------------------------------
# View Confusion matrix of classification model
help(crop_model.val_dataset_confusion)

label_dict = {
    "jute": 0,
    "maize": 1,
    "rice": 2,
    "sugarcane": 3,
    "wheat": 4
}

# Correctly unpack the 2 values returned by the function
true_class, predicted_class = crop_model.val_dataset_confusion(return_images=False)

print("True class values (first 10):", true_class[:10])
print("Predicted class values (first 10):", predicted_class[:10]) # Now these are already class IDs

# --- No need for np.argmax here, predicted_class is already the ID ---
y_pred = predicted_class # predicted_class already contains the class IDs

# Ensure true_class is also a NumPy array for consistency with sklearn functions
y_test = np.array(true_class)

# --- Extract class names in the correct order from your label_dict ---
sorted_labels = sorted(label_dict.items(), key=lambda item: item[1])
class_names = [label for label, _id in sorted_labels]
print(f"Class names for display: {class_names}")

# --- Display the confusion matrix with labels ---
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap='viridis')
plt.title("Confusion Matrix for Crop Model")
plt.show()
plt.close()


# ---------------------------------------
# Create RetinaNet detection trainer

# --- Path Definitions based on parameter ---
if multiperspective:
    print("Configuring for Multiperspective dataset...")
    cwd = "/content/data/uav_crops_data_multiperspective"
    main_model_images_folder = "crops_multiperspective_images"
    main_model_annotations_csv = "crops_multiperspective_detection_annotations.csv"
else:
    print("Configuring for Aerial dataset...")
    cwd = "/content/data/uav_crops_data_aerial"
    main_model_images_folder = "crops_aerial_images"
    main_model_annotations_csv = "crops_aerial_detection_annotations.csv"

main_model_path = os.path.join(cwd, main_model_images_folder)
main_model_csv_file_path = os.path.join(cwd, main_model_annotations_csv)

print(f"Base directory (cwd): {cwd}")
print(f"Main model images root directory: {main_model_path}")
print(f"Main model annotations CSV: {main_model_csv_file_path}")

# --- Configuration for RetinaNet object detection model ---
cfg = {}
cfg["train"] = {}
cfg["validation"] = {}
cfg["num_classes"] = 5 # As determined by your label_dict
cfg["train"]["fast_dev_run"] = False
# Reducing batch size for out of memory issue
#cfg["batch_size"] = 16
cfg["batch_size"] = 4
#cfg["workers"] = 16
cfg["workers"] = 8

# Set GPUs based on availability
if torch.cuda.is_available():
    cfg["gpus"] = '-1' # Use all available GPUs
    print("CUDA is available. Setting gpus to '-1'.")
else:
    cfg["gpus"] = '0' # Use CPU if no GPU
    print("CUDA is not available. Setting gpus to '0' (CPU).")

cfg["train"]["epochs"] = 10
cfg["train"]["learning_rate"] = 0.0001
print(f"Current learning rate: {cfg['train']['learning_rate']}")

# --- Assign the dynamically determined paths to cfg ---
cfg["train"]["csv_file"] = main_model_csv_file_path
cfg["train"]["root_dir"] = main_model_path
cfg["validation"]["csv_file"] = main_model_csv_file_path
cfg["validation"]["root_dir"] = main_model_path

# Set the score_thresh very low as we rely on cropmodel_score, which will be high
# Setting score_thresh high will eliminate the good matches from cropmodel_score
cfg["score_thresh"] = 0.001 # Or even lower, like 0.001 to test fine-tuning

# Initialize the DeepForest model
# Hardcode label_dict as per your previous successful loading
m = main.deepforest(config_args=cfg,
                    num_classes=5,
                    label_dict={
                        "jute": 0,
                        "maize": 1,
                        "rice": 2,
                        "sugarcane": 3,
                        "wheat": 4
                    })

print("Creating trainer")
# Create a PyTorch Lightning trainer used for training
m.create_trainer()
print("Created trainer")


# ---------------------------------------
# Run this for Fine-tuning the tree-crown model

# Get a copy of the pretrained model so that we can use parts of it in our new model
deepforest_release_model = main.deepforest()
deepforest_release_model.use_release()

# Replace the backbone of our new model with the pretrained backbone.
m.model.backbone.load_state_dict(deepforest_release_model.model.backbone.state_dict())

# Replace the regression head of the new model with the pretrained regression head.
m.model.head.regression_head.load_state_dict(deepforest_release_model.model.head.regression_head.state_dict())

# ---------------------------------------
# Train the model now

# Addressing out of memory issue.
# After a major step where a lot of memory might have been temporarily used
# e.g., after validation epoch, or between large processing blocks
torch.cuda.empty_cache()

start_time = time.time()
m.trainer.fit(m)
print ("Fitting deepforest BB trainer done")

m.trainer.validate(m)
print ("reating deepforest BB trainer done")

print(f"--- Training on GPU: {(time.time() - start_time):.2f} seconds ---")


# ---------------------------------------

# Save the trained models state dictionaries
import torch

# Define the base directory for the UAV crops data based on the parameter
if multiperspective:
    base_model_data_dir = "/content/data/uav_crops_data_multiperspective"
    print("Configuring to save models for Multiperspective dataset.")
else:
    base_model_data_dir = "/content/data/uav_crops_data_aerial"
    print("Configuring to save models for Aerial dataset.")

# Create the directory for trained models
trained_models_dir = os.path.join(base_model_data_dir, "trained_models")
os.makedirs(trained_models_dir, exist_ok=True) # Ensure the directory exists

global_file_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Save CropModel weights using standard PyTorch method
class_model_save_path = os.path.join(trained_models_dir, f"EcoDetect_Classification_{global_file_timestamp}.pt")

# Assuming 'crop_model' is an instantiated and trained PyTorch model
torch.save(crop_model.state_dict(), class_model_save_path)
print("\nClassification Model saved to:")
print(class_model_save_path)

# Save Fine-tuned / Trained DeepForest Detection Model
detection_model_save_path = os.path.join(trained_models_dir, f"EcoDetect_BB_{global_file_timestamp}.pt")

# Assuming 'm' is an instantiated and trained DeepForest model
m.save_model(detection_model_save_path) # DeepForest's main model has its own save_model method
print("\nDetection Model saved to:")
print(detection_model_save_path)

print("\nAll model saving operations complete.")


# ---------------------------------------

# View tensorboard graphs and results in colab

#%load_ext tensorboard
#%tensorboard --logdir /content/lightning_logs

# ---------------------------------------


# DEBUG
# Run this to inspect the label dict of the saved checkpoint files

# --- Define detection_model_path based on parameter ---
# NOTE: Replace 'YYYYMMDDHHMMSS' with the actual timestamp
# of the saved model file (e.g., EcoDetect_BB_20250716055312.pt)
if multiperspective:
    detection_model_path = "/content/data/uav_crops_data_multiperspective/trained_models/EcoDetect_BB_YYYYMMDDHHMMSS.pt"
    print("Configuring to inspect Multiperspective detection model.")
else:
    detection_model_path = "/content/data/uav_crops_data_aerial/trained_models/EcoDetect_BB_YYYYMMDDHHMMSS.pt"
    print("Configuring to inspect Aerial detection model.")

print(f"Attempting to load model from: {detection_model_path}")

try:
    loaded_ckpt = torch.load(detection_model_path, map_location='cpu')

    print('\nCheckpoint keys:', loaded_ckpt.keys())
    if 'state_dict' in loaded_ckpt:
        print('State_dict keys (first 10):', list(loaded_ckpt['state_dict'].keys())[:10])
        # Also print the shape of a key related to num_classes, e.g., the final classification layer weight
        cls_logits_key_candidates = [k for k in loaded_ckpt['state_dict'].keys() if 'cls_logits.weight' in k]
        if cls_logits_key_candidates:
            print(f"Shape of {cls_logits_key_candidates[0]}: {loaded_ckpt['state_dict'][cls_logits_key_candidates[0]].shape}")
        else:
            print("No 'cls_logits.weight' found in state_dict.")

    if 'label_dict' in loaded_ckpt:
        print('\nLabel dict:', loaded_ckpt['label_dict'])
        print('Length of Label dict:', len(loaded_ckpt['label_dict']))
    else:
        print("\nNo 'label_dict' found in checkpoint.")

except FileNotFoundError:
    print(f"\nError: Model file not found at {detection_model_path}. Please ensure the path and filename (including timestamp) are correct.")
except Exception as e:
    print(f"\nAn error occurred while loading or inspecting the model: {e}")


# ---------------------------------------

"""
# RUN ONLY IF LOADING SAVED TRAINED CHECKPOINTS
# Don't run this if training the models

if multiperspective:
    base_data_dir = "/content/data/uav_crops_data_multiperspective"
    print("Configuring to load models from Multiperspective dataset paths.")
else:
    base_data_dir = "/content/data/uav_crops_data_aerial"
    print("Configuring to load models from Aerial dataset paths.")

detection_model_path = os.path.join(base_data_dir, "trained_models", "EcoDetect_BB_YYYYMMDDHHMMSS.pt")
classification_model_path = os.path.join(base_data_dir, "trained_models", "EcoDetect_Classification_YYYYMMDDHHMMSS.pt")

print(f"Attempting to load Detection Model from: {detection_model_path}")
print(f"Attempting to load Classification Model from: {classification_model_path}")

# --- 1. Load the DeepForest Detection Model (main.deepforest) ---
print("\nLoading DeepForest Detection Model...")

try:
    checkpoint = torch.load(detection_model_path, map_location='cpu')

    # Get label_dict and num_classes directly from checkpoint
    if 'label_dict' in checkpoint and 'num_classes' in checkpoint:
        loaded_label_dict = checkpoint['label_dict']
        loaded_num_classes = checkpoint['num_classes']
        print(f"Using label_dict from checkpoint: {loaded_label_dict} (Length: {len(loaded_label_dict)})")
        print(f"Using num_classes from checkpoint: {loaded_num_classes}")
    else:
        # Fallback if label_dict/num_classes are not directly in checkpoint top-level
        print("Warning: 'label_dict' and/or 'num_classes' not found at top level of checkpoint. Using hardcoded values (if applicable) or default DeepForest behavior.")
        loaded_label_dict = {
            "jute": 0, "maize": 1, "rice": 2, "sugarcane": 3, "wheat": 4
        } # Example hardcoded
        loaded_num_classes = 5 # Example hardcoded


    # Extract the actual model state dictionary
    raw_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # --- Key Remapping Logic ---
    remapped_state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("model."):
            k_stripped = k[len("model."):]
        else:
            k_stripped = k

        if k_stripped.startswith(("conv", "bn", "layer", "downsample", "fc")):
            new_key = f"model.backbone.body.{k_stripped}"
        elif k_stripped.startswith("fpn."):
            new_key = f"model.backbone.{k_stripped}"
        elif k_stripped.startswith(("head.classification_head.", "head.regression_head.")):
            new_key = f"model.{k_stripped}"
        else:
            new_key = k
        remapped_state_dict[new_key] = v

    # --- Initialize the DeepForest model using config_args ---
    cfg = {} # Assuming cfg is defined or can be an empty dict
    m = main.deepforest(config_args=cfg,
                        num_classes=loaded_num_classes, # Use loaded_num_classes
                        label_dict=loaded_label_dict) # Use loaded_label_dict

    # Load the remapped state dictionary
    m.load_state_dict(remapped_state_dict, strict=False)
    print("State dictionary keys remapped and loaded (strict=False).")

    # Set DeepForest configuration parameters
    m.config["score_thresh"] = 0.01

    # Move model to GPU and set to evaluation mode
    if torch.cuda.is_available():
        m.to('cuda')
        print("DeepForest model moved to CUDA.")
    m.eval()
    print("DeepForest Detection Model loaded successfully.")

except FileNotFoundError:
    print(f"Error: DeepForest Detection Model file not found at {detection_model_path}. Please check the path and filename (including timestamp). ❌")
except Exception as e:
    print(f"Error loading DeepForest Detection Model: {e} ❌")
    print("\n--- Important Troubleshooting Steps for DeepForest ---")
    print("1.  **DeepForest Version Mismatch**: Ensure your installed DeepForest version is compatible with the saved checkpoint.")
    print("2.  **Inspect .pt File Contents**: Verify the checkpoint structure using the debug script we discussed earlier.")
    raise

# -------------------------------------------------------------------

# --- 2. Load the CropModel Classification Model ---
print("\nLoading CropModel Classification Model...")
try:
    # Assuming 5 classes for the CropModel Classification model based on prior code.
    crop_model = model.CropModel(num_classes=5)

    # Load the state_dict directly
    crop_model.load_state_dict(torch.load(classification_model_path, map_location='cpu'))

    if torch.cuda.is_available():
        crop_model.to('cuda')
        print("CropModel moved to CUDA.")
    crop_model.eval()
    print("CropModel Classification Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: CropModel Classification Model file not found at {classification_model_path}. Please check the path and filename (including timestamp). ❌")
except Exception as e:
    print(f"Error loading CropModel Classification Model: {e}")
    print("\n--- Important Troubleshooting Steps for CropModel ---")
    print("1.  **Model Class Definition**: Ensure your `CropModel` class definition matches the one used during training.")
    print("2.  **Checkpoint Structure**: Verify the structure of the saved state_dict.")
    raise

# -------------------------------------------------------------------

print("\nBoth models are loaded and ready to use. ✨")



"""

# -------------------------------------------------------------------

# Run predictions and save csv output files

# Assume 'm' is the initialized DeepForest model
# Assume 'global_file_timestamp' is already defined


# --- Define the new output directory based on the parameter ---
if multiperspective:
    base_data_dir = "/content/data/uav_crops_data_multiperspective"
    print("Configuring prediction output directory for Multiperspective dataset.")
else:
    base_data_dir = "/content/data/uav_crops_data_aerial"
    print("Configuring prediction output directory for Aerial dataset.")

pred_output_dir = os.path.join(base_data_dir, "pred_output")
os.makedirs(pred_output_dir, exist_ok=True) # Ensure the output directory exists

print(f"Prediction output directory set to: {pred_output_dir}")



# Assume 'global_file_timestamp' is already defined from a previous step
# If not, generate it here
global_file_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


print("################### Calling predict tile")

path_to_images = "/content/data/test_crop_image" # Directory containing images to predict on
ext = [".jfif", ".jpg", ".png", ".jpeg"]

# --- Use pred_output_dir for all outputs ---
output_data_dir = pred_output_dir

for root, subdirs, files in os.walk(path_to_images):
  for file_name in files:
    if file_name.endswith(tuple(ext)):
      file_path = os.path.join(root, file_name)

      print("\nfile_path #############")
      print(file_path)

      # Assuming get_data is a function that retrieves content ID or just returns the path for local files
      # If file_path is already a direct path, get_data might not be needed.
      raster_path = file_path # If get_data is for content IDs, replace with get_data(file_path)
      print("raster_path ############# ")
      print(raster_path)

      iou_threshold = 0.15
      mosaic = True
      patch_size = 400
      patch_overlap = 0.05

      result = None
      isBBMatched = False

      try:
          print("Attempting predict_tile with bounding box dataset...")
          result = m.predict_tile(path=raster_path, # Changed 'raster_path' to 'path'
                                  patch_size=patch_size,
                                  patch_overlap=patch_overlap,
                                  iou_threshold=iou_threshold,
                                  # DeepForest v1.x often does not take 'crop_model' directly here.
                                  # If your setup truly integrates it here, keep it. Otherwise, remove.
                                  crop_model=crop_model
                                 )
          isBBMatched = True
          print("######### Predict Tile done. result: ########")
          print(result)
      except TypeError as e:
          print(f"Predict with bounding box dataset resulted in TypeError (argument mismatch): {e}")
          traceback.print_exc(file=sys.stdout)
      except Exception as e:
          print(f"Predict with bounding box dataset resulted in no matches or other error: {e}")
          traceback.print_exc(file=sys.stdout)

      if not isBBMatched or (result is not None and result.empty): # Check for not None AND empty DataFrame
          try:
              print("Predicting match for the whole image (fallback)")
              result = m.predict_image(path=raster_path)
              print("######### Predict Image done. result: ########")
              print(result)
          except TypeError as e:
              print(f"Predict image failed for {raster_path} due to TypeError (argument mismatch): {e}")
              traceback.print_exc(file=sys.stdout)
          except Exception as e:
              print(f"Predict image failed for {raster_path} due to other error: {e}")
              traceback.print_exc(file=sys.stdout)

      if result is not None and not result.empty:
          df_result = pd.DataFrame(result)
          timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

          # --- Save files to output_data_dir (which is pred_output_dir) ---
          scores_output_file = os.path.join(output_data_dir, f"tiles_output_{timestamp}.csv")
          df_result.to_csv(scores_output_file, index=False, mode='a', header=False) # Use header=False for appends
          print(f"  Saved tiles_output to: {scores_output_file}")

          histogram_output_file = os.path.join(output_data_dir, f"hist_output_{global_file_timestamp}.csv")
          with open(histogram_output_file, "a") as hist_file:
              line = f"{file_path},{len(df_result)}"
              hist_file.write(line + '\n')
          print(f"  Appended to hist_output: {histogram_output_file}")
          # --- Saved files ---

          print("######### Predict Tile for df done. result: ########")
          print(df_result)
      else:
          print(f"No results or empty DataFrame to display for {file_name}.")

print("Prediction and file generation process complete.")