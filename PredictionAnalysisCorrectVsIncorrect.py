import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_prediction_accuracy_with_filters(file_path):
    """
    Reads a CSV file, filters predictions based on score and box size,
    then analyzes the accuracy of the remaining predictions within
    different cropmodel_score buckets and visualizes the results.

    Args:
        file_path (str): The path to the CSV file. The file must contain
                         'predicted_label', 'true_label', and 'cropmodel_score' columns,
                         as well as 'xmin', 'ymin', 'xmax', and 'ymax' for box dimensions.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the CSV file is uploaded to the Colab environment or the correct path is provided.")
        return

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Check for all required columns
        required_columns = ['predicted_label', 'true_label', 'cropmodel_score', 'xmin', 'ymin', 'xmax', 'ymax']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            print(f"Error: The following required column(s) were not found in '{file_path}': {', '.join(missing_cols)}")
            print("Please ensure the CSV file that contains these columns, such as 'pred_output_tiles_multiperspective_with_true_label.csv'.")
            return

        initial_row_count = len(df)
        print(f"Total number of predictions before filtering: {initial_row_count}")

        # --- Define and apply filters ---
        # Adjust these values to suit specific analysis needs.
        MIN_CROPMODEL_SCORE = 0.0
        MIN_BOX_WIDTH = 80
        MIN_BOX_HEIGHT = 80
        MIN_BOX_AREA = 6400  # Example: 80 * 80

        # Filter by minimum cropmodel_score
        df = df[df['cropmodel_score'] >= MIN_CROPMODEL_SCORE]
        filtered_by_score_count = len(df)
        print(f"  Predictions after filtering by score (>= {MIN_CROPMODEL_SCORE}): {filtered_by_score_count}")

        # Calculate box dimensions for filtering
        df['box_width'] = df['xmax'] - df['xmin']
        df['box_height'] = df['ymax'] - df['ymin']
        df['box_area'] = df['box_width'] * df['box_height']

        # Filter by minimum box dimensions
        df = df[df['box_width'] >= MIN_BOX_WIDTH]
        df = df[df['box_height'] >= MIN_BOX_HEIGHT]
        df = df[df['box_area'] >= MIN_BOX_AREA]

        filtered_by_size_count = len(df)
        print(f"  Predictions after filtering by size (W:>{MIN_BOX_WIDTH}, H:>{MIN_BOX_HEIGHT}, A:>{MIN_BOX_AREA}): {filtered_by_size_count}")

        if df.empty:
            print("\nNo predictions matched the filter criteria. Please adjust filter values and try again.")
            return

        # --- Analysis on the filtered data ---

        # Create a new column 'is_correct'
        df['is_correct'] = (df['predicted_label'] == df['true_label'])

        # Define score bins and labels for the buckets
        bins = np.arange(0, 1.1, 0.1)
        labels = [f'{b:.1f}-{b+0.1:.1f}' for b in bins[:-1]]

        # Create a new column 'score_bucket' by binning the scores
        df['score_bucket'] = pd.cut(df['cropmodel_score'], bins=bins, labels=labels, right=False)

        # Group by score bucket and calculate correct/incorrect counts
        accuracy_by_bucket = df.groupby('score_bucket')['is_correct'].value_counts().unstack(fill_value=0)

        # Ensure both True (correct) and False (incorrect) columns exist
        if True not in accuracy_by_bucket.columns:
            accuracy_by_bucket[True] = 0
        if False not in accuracy_by_bucket.columns:
            accuracy_by_bucket[False] = 0

        # Rename columns for clarity
        accuracy_by_bucket = accuracy_by_bucket.rename(columns={True: 'Correct Predictions', False: 'Incorrect Predictions'})

        # Calculate and add the percent correct column
        accuracy_by_bucket['Total Predictions'] = accuracy_by_bucket['Correct Predictions'] + accuracy_by_bucket['Incorrect Predictions']
        accuracy_by_bucket['Percent Correct'] = (accuracy_by_bucket['Correct Predictions'] / accuracy_by_bucket['Total Predictions']) * 100

        # Print a summary table
        print("\nAccuracy by Crop Model Score Bucket :")
        print(accuracy_by_bucket.to_markdown(floatfmt=".2f"))
        print("\n")

        # Create the bar chart
        accuracy_by_bucket[['Correct Predictions', 'Incorrect Predictions']].plot(kind='bar', figsize=(12, 7), stacked=True, color=['#4CAF50', '#F44336'])
        plt.title('Correct vs. Incorrect Predictions by Score Bucket', fontsize=16)
        plt.xlabel('Crop Model Score Bucket', fontsize=12)
        plt.ylabel('Number of Predictions', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Prediction Outcome')
        plt.tight_layout()
        plt.show()

    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Could not parse '{file_path}'. Please check if it's a valid CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Instructions for use in Colab ---
    # 1. Upload one of the CSV files with true labels to Colab environment.
    #    For example: 'pred_output_tiles_multiperspective_with_true_label.csv'
    # 2. Adjust the file_path and filter constants (MIN_CROPMODEL_SCORE, MIN_BOX_WIDTH, etc.) below.
    # 3. Run this script.

    #file_path = "pred_output_tiles_multiperspective_with_true_label.csv"
    file_path = "pred_output_tiles_aerial_with_true_label.csv"

    analyze_prediction_accuracy_with_filters(file_path)
