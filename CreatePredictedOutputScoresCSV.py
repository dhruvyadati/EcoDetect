# Create a final CSV file with all the scores by merging all the individual tiles output csv files
# This CSV file will be used to run analysis on accuracy scores of predictions.
import os
import pandas as pd



# Define the base directory and output filename based on the parameter
if multiperspective:
    base_data_dir = "/content/data/uav_crops_data_multiperspective"
    output_filename = "pred_output_tiles_multiperspective.csv"
    print("Configuring to merge tiles for Multiperspective dataset.")
else:
    base_data_dir = "/content/data/uav_crops_data_aerial"
    output_filename = "pred_output_tiles_aerial.csv"
    print("Configuring to merge tiles for Aerial dataset.")

pred_output_dir = os.path.join(base_data_dir, "pred_output")
output_file_path = os.path.join(base_data_dir, output_filename) # Save the merged file in base_data_dir

# Define the headers
CSV_HEADERS = [
    'xmin', 'ymin', 'xmax', 'ymax',
    'predicted_label_internal', 'predicted_score_internal',
    'image_path',
    'true_label',
    'cropmodel_score',
    'polygon_wkt'
]

# List to hold individual DataFrames
all_dfs = []

print(f"Scanning for tiles in: {pred_output_dir}")

# Iterate over files in the pred_output directory
for file_name in os.listdir(pred_output_dir):
    if file_name.startswith('tiles_output_') and file_name.endswith('.csv'):
        file_path = os.path.join(pred_output_dir, file_name)
        print(f"  Reading: {file_name}")
        try:
            # Read the CSV file without a header, then assign custom headers
            df = pd.read_csv(file_path, header=None, names=CSV_HEADERS)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error reading {file_name}: {e}. Skipping this file.")

if not all_dfs:
    print(f"No 'tiles_output_*.csv' files found in '{pred_output_dir}'. No merge performed.")
else:
    # Concatenate all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file_path, index=False)
    print(f"\nSuccessfully merged {len(all_dfs)} CSV files into: {output_file_path}")
    print(f"Total rows in merged file: {len(merged_df)}")