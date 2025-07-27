# Run this if we want to visualize previously predicted and archived image files
# Unzip the previously loaded images into pred_output directory

import os
import zipfile

import matplotlib.pyplot as plt
import cv2
import supervision as sv
import pandas as pd
import numpy as np


# Define global variable multiperspective
multiperspective = True # Assuming multiperspective is the default based on previous interactions

# Define the zip file name
zip_file = "/content/pred_output_archive.zip"

# Determine the correct destination directory based on the multiperspective flag
if multiperspective:
    destination_base_dir = "/content/data/uav_crops_data_multiperspective"
    print(f"Preparing to unzip {zip_file} into multiperspective directory.")
else:
    destination_base_dir = "/content/data/uav_crops_data_aerial"
    print(f"Preparing to unzip {zip_file} into aerial directory.")

# Unzip into the 'pred_output' subdirectory
destination_dir = os.path.join(destination_base_dir, "pred_output")

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)
print(f"Destination directory for unzipping: {destination_dir}")

# Construct and print the unzip command. Using the 'zipfile' module for direct Python unzipping
# as shell commands can sometimes have environment-specific issues.
print(f"Attempting to unzip {zip_file} to {destination_dir} using Python's zipfile module.")

try:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)
    print(f"Successfully unzipped {zip_file} to {destination_dir}.")
except FileNotFoundError:
    print(f"Error: The zip file '{zip_file}' was not found. Please ensure it is in the correct directory.")
except zipfile.BadZipFile:
    print(f"Error: '{zip_file}' is not a valid zip file or is corrupted.")
except Exception as e:
    print(f"An unexpected error occurred during unzipping: {e}")


# -------------------------------------------------------------------


# --- Configuration ---
# Define the base directory for your UAV crops data outputs based on the parameter
if multiperspective:
    base_data_dir = "/content/data/uav_crops_data_multiperspective"
    print("Setting base data directory for Multiperspective dataset.")
else:
    base_data_dir = "/content/data/uav_crops_data_aerial"
    print("Setting base data directory for Aerial dataset.")

print(f"Base data directory set to: {base_data_dir}")

# Directory where tiles_output_*.csv files are located
csv_output_dir = os.path.join(base_data_dir, "pred_output")

# --- Directory where the original images are located ---
image_source_dir = "/content/data/test_crop_image"

# Directory for storing the generated bounding-boxed images
pred_boxed_images_dir = os.path.join(base_data_dir, "pred_bounding_boxed_images")
os.makedirs(pred_boxed_images_dir, exist_ok=True) # Ensure this directory exists

# Define the headers expected in your CSV.
CSV_HEADERS = [
    'xmin', 'ymin', 'xmax', 'ymax',
    'predicted_label_internal', 'predicted_score_internal',
    'image_path',
    'true_label',
    'cropmodel_score',
    'polygon_wkt'
]

# Define the minimum cropmodel_score for visualization
MIN_CROPMODEL_SCORE = 0.9

# --- Define minimum dimensions/area for boxes to keep ---
# Adjust these values based on your observations of the unwanted smaller boxes.
# For example, if small boxes are typically less than 20x20 pixels or have an area less than 400.
MIN_BOX_WIDTH = 80
MIN_BOX_HEIGHT = 80
MIN_BOX_AREA = 6400 # Example 400: boxes must have an area of at least 400 square pixels (20*20)

# --- STATIC COLOR PALETTE CONFIGURATION ---
# Define a fixed mapping from crop names to a static integer ID
# The order of this list determines the class_id (index) and thus the color.
# Ensure this list covers all possible 'true_label' values from your data.
FIXED_CROP_NAMES = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']
FIXED_CLASS_NAME_TO_ID = {name: i for i, name in enumerate(FIXED_CROP_NAMES)}

# Define specific bright colors for each class in the order of FIXED_CROP_NAMES
FIXED_CUSTOM_COLORS = [
    sv.Color.from_hex("#FFFF00"), # Bright Yellow (e.g., for jute)
    sv.Color.from_hex("#FFA500"), # Orange (e.g., for maize)
    sv.Color.from_hex("#39FF14"), # Neon Green (e.g., for rice)
    sv.Color.from_hex("#00FFFF"), # Cyan (e.g., for sugarcane)
    sv.Color.from_hex("#FF00FF")  # Magenta (e.g., for wheat)
]
# Create a custom ColorPalette from the fixed list of colors
color_palette = sv.ColorPalette(colors=FIXED_CUSTOM_COLORS)


# --- Main Visualization Logic ---
print(f"Scanning for CSV files in: {csv_output_dir}")
print(f"Original images source directory: {image_source_dir}")
print(f"Generated bounding boxed images will be stored in: {pred_boxed_images_dir}")

# Loop through all files in the specified CSV directory
for file_name in os.listdir(csv_output_dir):
    if file_name.startswith('tiles_output_') and file_name.endswith('.csv'):
        csv_file_path = os.path.join(csv_output_dir, file_name)
        print(f"\nProcessing CSV file: {csv_file_path}")

        try:
            # Read the CSV, explicitly providing headers. No header in the file itself.
            df_predictions = pd.read_csv(csv_file_path, header=None, names=CSV_HEADERS)

            # Ensure numeric columns are actually numeric
            for col in ['xmin', 'ymin', 'xmax', 'ymax', 'cropmodel_score']:
                df_predictions[col] = pd.to_numeric(df_predictions[col], errors='coerce')

            # Drop rows where essential numeric columns are NaN after coercion
            df_predictions.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax', 'cropmodel_score'], inplace=True)

            # Convert bounding box coordinates to integers
            df_predictions[['xmin', 'ymin', 'xmax', 'ymax']] = df_predictions[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)

            # --- Apply the score filter ---
            initial_count = len(df_predictions)
            df_predictions = df_predictions[df_predictions['cropmodel_score'] >= MIN_CROPMODEL_SCORE]
            filtered_by_score_count = len(df_predictions)
            print(f"  Filtered predictions: {initial_count} initially, {filtered_by_score_count} after score threshold {MIN_CROPMODEL_SCORE}.")

            # Check if the DataFrame is empty after filtering by score
            if df_predictions.empty:
                print(f"  No detections found above score threshold {MIN_CROPMODEL_SCORE} in {file_name}. Skipping visualization.")
                continue

            # --- Apply the size filter logic ---
            df_predictions['box_width'] = df_predictions['xmax'] - df_predictions['xmin']
            df_predictions['box_height'] = df_predictions['ymax'] - df_predictions['ymin']
            df_predictions['box_area'] = df_predictions['box_width'] * df_predictions['box_height']

            # Apply the size filter (you can combine these or use just one/two)
            df_predictions = df_predictions[df_predictions['box_width'] >= MIN_BOX_WIDTH]
            df_predictions = df_predictions[df_predictions['box_height'] >= MIN_BOX_HEIGHT]
            df_predictions = df_predictions[df_predictions['box_area'] >= MIN_BOX_AREA]

            filtered_by_size_count = len(df_predictions)
            print(f"  Filtered predictions by size: {filtered_by_score_count} initially, {filtered_by_size_count} after size filter (min W:{MIN_BOX_WIDTH}, H:{MIN_BOX_HEIGHT}, Area:{MIN_BOX_AREA}).")


            # Check if the DataFrame is empty after filtering by size
            if df_predictions.empty:
                print(f"  No detections found above size thresholds in {file_name}. Skipping visualization.")
                continue

            # Get the unique original image name from the CSV
            unique_images = df_predictions['image_path'].unique()
            if len(unique_images) > 1:
                print(f"  Warning: Multiple unique images found in {file_name}. Visualizing for the first: {unique_images[0]}")

            image_name = unique_images[0]
            original_image_full_path = os.path.join(image_source_dir, image_name)

            if not os.path.exists(original_image_full_path):
                print(f"  Error: Original image '{original_image_full_path}' not found. Skipping visualization for this CSV.")
                continue

            # Load the original image
            image_bgr = cv2.imread(original_image_full_path)
            if image_bgr is None:
                print(f"  Error: Could not load image '{original_image_full_path}'. Skipping visualization.")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # --- Use the FIXED class mapping for coloring ---
            # Map the 'true_label' column to the fixed class_id
            class_ids = df_predictions['true_label'].map(FIXED_CLASS_NAME_TO_ID).values.astype(int)

            # Convert pandas DataFrame to Supervision Detections format
            boxes = df_predictions[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            confidence = df_predictions['cropmodel_score'].values.astype(np.float32) # Use cropmodel_score for confidence
            labels = df_predictions['true_label'].values.astype(str) # Use 'true_label' for display label

            sv_detections = sv.Detections(
                xyxy=boxes,
                confidence=confidence,
                class_id=class_ids, # Use the FIXED mapped class_ids for consistent coloring
                data={'class_name': labels}
            )

            # Initialize annotators with thinner lines and custom colors
            box_annotator = sv.BoxAnnotator(
                thickness=1, # Thinner lines
                color=color_palette # Use the custom color palette
            )
            label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.BOTTOM_CENTER,
                text_scale=0.6, # Slightly smaller text for sharper look
                text_thickness=1, # Thinner text
                text_padding=5,
                text_color=sv.Color.BLACK, # Set label text to black
                color=color_palette # Use the custom color palette for label background
            )

            # Annotate the image
            annotated_frame = box_annotator.annotate(
                scene=image_rgb.copy(),
                detections=sv_detections
            )

            # Prepare labels for the label annotator: "label confidence_score"
            display_labels = [
                f"{class_name} {score:.2f}"
                for class_name, score in zip(sv_detections.data['class_name'], sv_detections.confidence)
            ]

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=sv_detections,
                labels=display_labels
            )

            # Display the annotated image
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_frame)
            plt.title(f"Detections (Score >= {MIN_CROPMODEL_SCORE}) on {image_name}")
            plt.axis('off')

            # --- Save annotated image to pred_boxed_images_dir ---
            output_image_filename = f"annotated_{os.path.splitext(image_name)[0]}_score{MIN_CROPMODEL_SCORE}.jpg"
            output_image_path = os.path.join(pred_boxed_images_dir, output_image_filename)
            plt.savefig(output_image_path)
            plt.close()
            print(f"  Annotated image saved to: {output_image_path}")

        except pd.errors.ParserError as pe:
            print(f"  Error parsing CSV '{csv_file_path}': {pe}. Please check CSV format.")
        except Exception as e:
            print(f"  An unexpected error occurred processing {csv_file_path}: {e}")

print("\nVisualization process complete.")



