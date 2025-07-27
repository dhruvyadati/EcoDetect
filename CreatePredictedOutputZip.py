# Zip the predicted output files so we can download
# Also, remember to download the trained models

import shutil
import os

# Define the base directory where folders are located
if multiperspective:
    base_dir = "/content/data/uav_crops_data_multiperspective"
    print("Setting base data directory for Multiperspective dataset.")
else:
    base_dir = "/content/data/uav_crops_data_aerial"
    print("Setting base data directory for Aerial dataset.")

print(f"Base data directory set to: {base_data_dir}")

# Define the paths to the folders you want to zip
pred_output_folder = os.path.join(base_dir, "pred_output")
pred_images_folder = os.path.join(base_dir, "pred_bounding_boxed_images")

# Define the output path for the zip files
# The .zip extension will be added automatically
output_zip_name_pred_output = os.path.join(base_dir, "pred_output_archive")
output_zip_name_pred_images = os.path.join(base_dir, "pred_bounding_boxed_images_archive")

print(f"Zipping '{pred_output_folder}' to '{output_zip_name_pred_output}.zip'...")
try:
    shutil.make_archive(output_zip_name_pred_output, 'zip', pred_output_folder)
    print("pred_output folder zipped successfully!")
except FileNotFoundError:
    print(f"Error: Folder not found at {pred_output_folder}. Please ensure it exists.")
except Exception as e:
    print(f"An error occurred while zipping pred_output: {e}")

print(f"\nZipping '{pred_images_folder}' to '{output_zip_name_pred_images}.zip'...")
try:
    shutil.make_archive(output_zip_name_pred_images, 'zip', pred_images_folder)
    print("pred_bounding_boxed_images folder zipped successfully!")
except FileNotFoundError:
    print(f"Error: Folder not found at {pred_images_folder}. Please ensure it exists.")
except Exception as e:
    print(f"An error occurred while zipping pred_bounding_boxed_images: {e}")

print("\nZipping process complete. You can find the .zip files in your base directory.")
