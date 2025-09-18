# CLASSIFICATION ACCURACY ANALYSIS FOR ***MULTI-PERSPECTIVE OR AERIAL*** PREDICTIONS
# Change FILE PATH to point to prediction output file for multi-perspective or aerial

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_crop_classification_performance(file_path, use_old_filters=True):
    """
    Analyzes crop classification performance with comprehensive metrics and visualizations.

    Args:
        file_path (str): Path to the CSV file with predictions
        use_old_filters (bool): If True, uses stricter filters (80x80 boxes, score>=0.0)
                               If False, uses lenient filters (20x20 boxes, score>=0.5)
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check required columns
        required_columns = ['predicted_label', 'true_label', 'cropmodel_score', 'xmin', 'ymin', 'xmax', 'ymax']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            print(f"Error: Missing columns: {', '.join(missing_cols)}")
            return

        # Remove unknown labels - focus on 5 crop classes
        valid_labels = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']
        df = df[df['true_label'].isin(valid_labels)].copy()

        print(f"Total predictions (5 crop classes): {len(df)}")
        print("Crop classes: jute (0), maize (1), rice (2), sugarcane (3), wheat (4)")

        # Apply filters based on choice
        if use_old_filters:
            # Old script filters (stricter)
            MIN_CROPMODEL_SCORE = 0.0
            MIN_BOX_WIDTH = 80
            MIN_BOX_HEIGHT = 80
            MIN_BOX_AREA = 6400
            filter_description = "80x80 Minimum Box Size"
        else:
            # New script filters (more lenient)
            MIN_CROPMODEL_SCORE = 0.0
            MIN_BOX_WIDTH = 20
            MIN_BOX_HEIGHT = 20
            MIN_BOX_AREA = 400
            filter_description = "20x20 Minimum Box Size"

        print(f"\nApplying {filter_description}:")
        print(f"  Score >= {MIN_CROPMODEL_SCORE}")
        print(f"  Box size >= {MIN_BOX_WIDTH}x{MIN_BOX_HEIGHT} (area >= {MIN_BOX_AREA})")

        # Calculate box dimensions
        df['box_width'] = df['xmax'] - df['xmin']
        df['box_height'] = df['ymax'] - df['ymin']
        df['box_area'] = df['box_width'] * df['box_height']

        # Apply filters
        df_filtered = df[
            (df['cropmodel_score'] >= MIN_CROPMODEL_SCORE) &
            (df['box_width'] >= MIN_BOX_WIDTH) &
            (df['box_height'] >= MIN_BOX_HEIGHT) &
            (df['box_area'] >= MIN_BOX_AREA)
        ].copy()

        print(f"Predictions after filtering: {len(df_filtered)}")

        if df_filtered.empty:
            print("No predictions match the filter criteria!")
            return

        # Calculate correctness
        df_filtered['is_correct'] = (df_filtered['predicted_label'] == df_filtered['true_label'])
        overall_accuracy = df_filtered['is_correct'].mean() * 100
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

        # --- SCORE BUCKET ANALYSIS ---
        bins = np.arange(0, 1.1, 0.1)
        labels = [f'{b:.1f}-{b+0.1:.1f}' for b in bins[:-1]]
        df_filtered['score_bucket'] = pd.cut(df_filtered['cropmodel_score'], bins=bins, labels=labels, right=False)

        # Overall accuracy by score bucket
        print(f"\n{'='*60}")
        print("ACCURACY BY SCORE BUCKET (Overall)")
        print(f"{'='*60}")

        bucket_analysis = df_filtered.groupby('score_bucket')['is_correct'].agg(['count', 'sum', 'mean'])
        bucket_analysis.columns = ['Total_Predictions', 'Correct_Predictions', 'Accuracy_Rate']
        bucket_analysis['Incorrect_Predictions'] = bucket_analysis['Total_Predictions'] - bucket_analysis['Correct_Predictions']
        bucket_analysis['Percent_Correct'] = (bucket_analysis['Accuracy_Rate'] * 100).round(2)

        # Reorder columns for clarity
        bucket_analysis = bucket_analysis[['Correct_Predictions', 'Incorrect_Predictions', 'Total_Predictions', 'Percent_Correct']]
        print(bucket_analysis)

        # --- CLASS-SPECIFIC ANALYSIS ---
        print(f"\n{'='*60}")
        print("ACCURACY BY CROP CLASS")
        print(f"{'='*60}")
        print("Note: This shows how well each TRUE crop class was classified")
        print("(i.e., when the actual crop was X, how often was it correctly predicted as X)")

        class_accuracy = df_filtered.groupby('true_label')['is_correct'].agg(['count', 'sum', 'mean'])
        class_accuracy.columns = ['Total_Predictions', 'Correct_Predictions', 'Accuracy_Rate']
        class_accuracy['Incorrect_Predictions'] = class_accuracy['Total_Predictions'] - class_accuracy['Correct_Predictions']
        class_accuracy['Percent_Correct'] = (class_accuracy['Accuracy_Rate'] * 100).round(1)
        class_accuracy = class_accuracy[['Total_Predictions', 'Correct_Predictions', 'Incorrect_Predictions', 'Percent_Correct']]
        print(class_accuracy)

        # --- CONFUSION MATRIX ---
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX")
        print(f"{'='*60}")
        print("Rows = True Label, Columns = Predicted Label")
        confusion_matrix = pd.crosstab(df_filtered['true_label'], df_filtered['predicted_label'], margins=True)
        print(confusion_matrix)

        # --- VISUALIZATIONS ---
        plt.figure(figsize=(16, 12))

        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 3, 1)
        confusion_no_margins = pd.crosstab(df_filtered['true_label'], df_filtered['predicted_label'])
        sns.heatmap(confusion_no_margins, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=12)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # 2. Accuracy by Score Bucket
        plt.subplot(2, 3, 2)
        bucket_analysis[['Correct_Predictions', 'Incorrect_Predictions']].plot(kind='bar', stacked=True,
                                                                               color=['#4CAF50', '#F44336'], ax=plt.gca())
        plt.title('Predictions by Score Bucket', fontsize=12)
        plt.xlabel('Score Bucket')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(['Correct', 'Incorrect'])

        # 3. Accuracy Percentage by Score Bucket
        plt.subplot(2, 3, 3)
        bucket_analysis['Percent_Correct'].plot(kind='bar', color='skyblue')
        plt.title('Accuracy % by Score Bucket', fontsize=12)
        plt.xlabel('Score Bucket')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # 4. Accuracy by Crop Class
        plt.subplot(2, 3, 4)
        class_accuracy['Percent_Correct'].plot(kind='bar', color='lightcoral')
        plt.title('Prediction Accuracy % by True Crop Class', fontsize=12)
        plt.xlabel('True Label')
        plt.ylabel('Prediction Accuracy (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # 5. Score Distribution by Crop Class
        plt.subplot(2, 3, 5)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, label in enumerate(valid_labels):
            label_scores = df_filtered[df_filtered['true_label'] == label]['cropmodel_score']
            if len(label_scores) > 0:
                plt.hist(label_scores, bins=15, alpha=0.7, label=f'{label}',
                        color=colors[i], density=True)
        plt.xlabel('Crop Model Score')
        plt.ylabel('Density')
        plt.title('Score Distribution by Crop Class')
        plt.legend()
        plt.grid(alpha=0.3)

        # 6. Sample Count by Class
        plt.subplot(2, 3, 6)
        class_counts = df_filtered['true_label'].value_counts()
        class_counts.plot(kind='bar', color='lightgreen')
        plt.title('Sample Count by Crop Class', fontsize=12)
        plt.xlabel('True Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        plt.suptitle(f'Crop Classification Analysis - {filter_description}', fontsize=16)
        plt.tight_layout()
        plt.show()

        # --- DETAILED CLASS PERFORMANCE ---
        print(f"\n{'='*60}")
        print("DETAILED MISCLASSIFICATION ANALYSIS")
        print(f"{'='*60}")

        for label in valid_labels:
            label_df = df_filtered[df_filtered['true_label'] == label]
            if not label_df.empty:
                accuracy = label_df['is_correct'].mean() * 100
                total = len(label_df)
                correct = label_df['is_correct'].sum()

                print(f"\n{label.upper()}:")
                print(f"  Total: {total}, Correct: {correct}, Accuracy: {accuracy:.1f}%")

                # Show misclassifications
                misclassified = label_df[label_df['is_correct'] == False]
                if len(misclassified) > 0:
                    misclass_counts = misclassified['predicted_label'].value_counts()
                    print(f"  Misclassified as:")
                    for pred_label, count in misclass_counts.items():
                        pct = (count / total) * 100
                        print(f"    → {pred_label}: {count} times ({pct:.1f}%)")

        # --- COMPARISON WITH EXPECTED RESULTS ---
        print(f"\n{'='*60}")
        print("VERIFICATION: High-confidence predictions (score >= 0.9)")
        print(f"{'='*60}")

        high_conf = df_filtered[df_filtered['cropmodel_score'] >= 0.9]
        if len(high_conf) > 0:
            high_conf_accuracy = high_conf['is_correct'].mean() * 100
            print(f"Score >= 0.9: {high_conf['is_correct'].sum()}/{len(high_conf)} = {high_conf_accuracy:.1f}% accuracy")

        med_conf = df_filtered[(df_filtered['cropmodel_score'] >= 0.8) & (df_filtered['cropmodel_score'] < 0.9)]
        if len(med_conf) > 0:
            med_conf_accuracy = med_conf['is_correct'].mean() * 100
            print(f"Score 0.8-0.9: {med_conf['is_correct'].sum()}/{len(med_conf)} = {med_conf_accuracy:.1f}% accuracy")

    except Exception as e:
        print(f"Error occurred: {e}")

def compare_filter_effects(file_path):
    """
    Compare the effects of different filter settings to understand discrepancies.
    """
    print("COMPARING FILTER EFFECTS")
    print("="*50)

    # Run both filter settings
    print("1. OLD FILTERS (Stricter - matches previous results):")
    analyze_crop_classification_performance(file_path, use_old_filters=True)

    print("\n" + "="*50)
    print("2. NEW FILTERS (More Lenient):")
    analyze_crop_classification_performance(file_path, use_old_filters=False)

if __name__ == "__main__":
    file_path = "pred_output_tiles_multiperspective_with_true_label.csv"

    # Use old filters to match expected results
    analyze_crop_classification_performance(file_path, use_old_filters=True)

    # Uncomment below to compare both filter settings
    # compare_filter_effects(file_path)
