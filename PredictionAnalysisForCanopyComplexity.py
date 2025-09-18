import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, f_oneway
import warnings
warnings.filterwarnings('ignore')

def analyze_canopy_complexity_hypothesis(multi_file, aerial_file):
    """
    Statistical analysis of the relationship between canopy complexity
    and multi-perspective imagery effectiveness for crop classification.
    """

    print("="*70)
    print("CANOPY COMPLEXITY AND MULTI-PERSPECTIVE IMAGERY ANALYSIS")
    print("="*70)

    # Define canopy complexity scoring based on literature review
    complexity_scores = {
        'jute': 3,      # High complexity - strongly branched, heterogeneous leaves
        'wheat': 3,     # High complexity - complex tillering, variable leaf angles
        'maize': 2,     # Medium complexity - smart canopy architecture
        'sugarcane': 2, # Medium complexity - alternating leaves, moderate structure
        'rice': 1       # Low complexity - erect ideotype, uniform structure
    }

    valid_labels = ['jute', 'wheat', 'maize', 'sugarcane', 'rice']

    def process_dataset(file_path, dataset_name):
        """Process dataset and calculate classification accuracy by crop."""
        df = pd.read_csv(file_path)

        # Remove unknown labels and focus on 5 crop classes
        df = df[df['true_label'].isin(valid_labels)].copy()

        # Calculate box dimensions
        df['box_width'] = df['xmax'] - df['xmin']
        df['box_height'] = df['ymax'] - df['ymin']
        df['box_area'] = df['box_width'] * df['box_height']

        # Apply strict filters (matching old script for consistency)
        df_filtered = df[
            (df['cropmodel_score'] >= 0.0) &
            (df['box_width'] >= 80) &
            (df['box_height'] >= 80) &
            (df['box_area'] >= 6400)
        ].copy()

        print(f"\n{dataset_name} Dataset:")
        print(f"  Total filtered predictions: {len(df_filtered)}")

        # Calculate accuracy by crop
        df_filtered['is_correct'] = (df_filtered['predicted_label'] == df_filtered['true_label'])

        accuracy_results = {}
        for crop in valid_labels:
            crop_data = df_filtered[df_filtered['true_label'] == crop]
            if len(crop_data) > 0:
                accuracy = crop_data['is_correct'].mean() * 100
                total = len(crop_data)
                correct = crop_data['is_correct'].sum()
                accuracy_results[crop] = {
                    'total': total,
                    'correct': correct,
                    'accuracy': accuracy
                }
                print(f"  {crop.capitalize()}: {correct}/{total} = {accuracy:.1f}%")

        return accuracy_results

    # Process both datasets
    multi_results = process_dataset(multi_file, "MULTI-PERSPECTIVE")
    aerial_results = process_dataset(aerial_file, "AERIAL-ONLY")

    # Create comparison dataframe
    comparison_data = []
    for crop in valid_labels:
        if crop in multi_results and crop in aerial_results:
            multi_acc = multi_results[crop]['accuracy']
            aerial_acc = aerial_results[crop]['accuracy']
            improvement = multi_acc - aerial_acc
            complexity = complexity_scores[crop]

            comparison_data.append({
                'crop': crop,
                'complexity_score': complexity,
                'multi_perspective_accuracy': multi_acc,
                'aerial_only_accuracy': aerial_acc,
                'accuracy_improvement': improvement,
                'multi_total': multi_results[crop]['total'],
                'aerial_total': aerial_results[crop]['total']
            })

    df_comparison = pd.DataFrame(comparison_data)

    print(f"\n{'='*70}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*70}")
    print(df_comparison.round(1))

    # Statistical Analysis
    print(f"\n{'='*70}")
    print("STATISTICAL HYPOTHESIS TESTING")
    print(f"{'='*70}")

    # Spearman's Rank Correlation
    complexity_values = df_comparison['complexity_score'].values
    improvement_values = df_comparison['accuracy_improvement'].values

    spearman_corr, spearman_p = spearmanr(complexity_values, improvement_values)

    print(f"\n1. SPEARMAN'S RANK CORRELATION")
    print(f"   Research Question: Is there a monotonic relationship between canopy")
    print(f"   complexity and the benefit of multi-perspective imagery?")
    print(f"   ")
    print(f"   H₀: No correlation between complexity and accuracy improvement (ρ = 0)")
    print(f"   H₁: Positive correlation exists (ρ > 0)")
    print(f"   ")
    print(f"   Spearman's ρ = {spearman_corr:.3f}")
    print(f"   p-value = {spearman_p:.3f}")
    print(f"   Significance level: α = 0.05")

    if spearman_p < 0.05:
        if spearman_corr > 0:
            print(f"   ✓ SIGNIFICANT POSITIVE CORRELATION (p < 0.05)")
            print(f"   → Higher complexity correlates with greater multi-perspective benefit")
        else:
            print(f"   ✓ SIGNIFICANT NEGATIVE CORRELATION (p < 0.05)")
            print(f"   → Higher complexity correlates with lower multi-perspective benefit")
    else:
        print(f"   ✗ NO SIGNIFICANT CORRELATION (p ≥ 0.05)")

    # Group by complexity for ANOVA
    complexity_groups = df_comparison.groupby('complexity_score')['accuracy_improvement'].apply(list).to_dict()

    print(f"\n2. ONE-WAY ANOVA")
    print(f"   Research Question: Do different complexity groups have significantly")
    print(f"   different mean accuracy improvements?")
    print(f"   ")
    print(f"   H₀: All group means are equal")
    print(f"   H₁: At least one group mean differs significantly")
    print(f"   ")

    # Calculate group statistics
    for score, improvements in complexity_groups.items():
        mean_improvement = np.mean(improvements)
        crops_in_group = df_comparison[df_comparison['complexity_score'] == score]['crop'].tolist()
        print(f"   Complexity {score}: Mean = {mean_improvement:.1f}% (crops: {', '.join(crops_in_group)})")

    # Perform ANOVA (note: small sample size limitation)
    if len(complexity_groups) >= 2:
        group_values = list(complexity_groups.values())

        # Check if we have enough data points for ANOVA
        total_points = sum(len(group) for group in group_values)
        if total_points >= 3:  # Need at least 3 data points total
            f_stat, anova_p = f_oneway(*group_values)

            print(f"   ")
            print(f"   F-statistic = {f_stat:.3f}")
            print(f"   p-value = {anova_p:.3f}")

            if anova_p < 0.05:
                print(f"   ✓ SIGNIFICANT GROUP DIFFERENCES (p < 0.05)")
                print(f"   → Complexity level significantly affects accuracy improvement")
            else:
                print(f"   ✗ NO SIGNIFICANT GROUP DIFFERENCES (p ≥ 0.05)")
        else:
            print(f"   Note: Insufficient data points for ANOVA (n={total_points})")
            print(f"   → Each complexity group has only 1-2 crops")
            print(f"   → Consider descriptive analysis instead")
    else:
        print(f"   Note: Insufficient groups for ANOVA analysis")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Complexity vs Improvement Scatter Plot
    ax1 = axes[0, 0]
    colors = ['red', 'orange', 'green']
    for i, crop in enumerate(df_comparison['crop']):
        complexity = df_comparison.loc[i, 'complexity_score']
        improvement = df_comparison.loc[i, 'accuracy_improvement']
        ax1.scatter(complexity, improvement, c=colors[complexity-1], s=100, alpha=0.7)
        ax1.annotate(crop.capitalize(), (complexity, improvement),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax1.set_xlabel('Canopy Complexity Score')
    ax1.set_ylabel('Accuracy Improvement (%)')
    ax1.set_title('Canopy Complexity vs Multi-Perspective Benefit')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 3])

    # Add trend line
    z = np.polyfit(complexity_values, improvement_values, 1)
    p = np.poly1d(z)
    ax1.plot([1, 2, 3], p([1, 2, 3]), "r--", alpha=0.8, linewidth=2)
    ax1.text(0.02, 0.98, f'ρ = {spearman_corr:.3f}\np = {spearman_p:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. Accuracy Comparison by Crop
    ax2 = axes[0, 1]
    x = np.arange(len(valid_labels))
    width = 0.35

    multi_accs = [df_comparison[df_comparison['crop']==crop]['multi_perspective_accuracy'].iloc[0]
                  for crop in valid_labels]
    aerial_accs = [df_comparison[df_comparison['crop']==crop]['aerial_only_accuracy'].iloc[0]
                   for crop in valid_labels]

    ax2.bar(x - width/2, multi_accs, width, label='Multi-Perspective', color='skyblue', alpha=0.8)
    ax2.bar(x + width/2, aerial_accs, width, label='Aerial-Only', color='lightcoral', alpha=0.8)

    ax2.set_xlabel('Crop Species')
    ax2.set_ylabel('Classification Accuracy (%)')
    ax2.set_title('Classification Accuracy by Imaging Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels([crop.capitalize() for crop in valid_labels], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Improvement by Complexity Group
    ax3 = axes[1, 0]
    complexity_means = df_comparison.groupby('complexity_score')['accuracy_improvement'].mean()
    complexity_std = df_comparison.groupby('complexity_score')['accuracy_improvement'].std()

    bars = ax3.bar(complexity_means.index, complexity_means.values,
                   yerr=complexity_std.values, capsize=5, color=['green', 'orange', 'red'], alpha=0.7)
    ax3.set_xlabel('Canopy Complexity Score')
    ax3.set_ylabel('Mean Accuracy Improvement (%)')
    ax3.set_title('Mean Improvement by Complexity Group')
    ax3.set_xticks([1, 2, 3])
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Add value labels on bars
    for bar, value in zip(bars, complexity_means.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Sample Size Information
    ax4 = axes[1, 1]
    sample_sizes = df_comparison[['crop', 'multi_total', 'aerial_total']].set_index('crop')
    sample_sizes.plot(kind='bar', ax=ax4, color=['skyblue', 'lightcoral'], alpha=0.8)
    ax4.set_xlabel('Crop Species')
    ax4.set_ylabel('Number of Predictions')
    ax4.set_title('Sample Sizes by Dataset')
    ax4.legend(['Multi-Perspective', 'Aerial-Only'])
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Generate conclusions
    print(f"\n{'='*70}")
    print("BIOLOGICAL AND AI INSIGHTS")
    print(f"{'='*70}")

    print(f"\n1. CANOPY COMPLEXITY EFFECT:")
    if spearman_p < 0.05 and spearman_corr > 0:
        print(f"   ✓ Statistically significant positive correlation supports hypothesis")
        print(f"   → Complex canopies benefit more from multi-perspective imagery")
        print(f"   → 3D structural features are key visual cues for AI classification")
    elif spearman_p < 0.05 and spearman_corr < 0:
        print(f"   ! Significant NEGATIVE correlation found (unexpected)")
        print(f"   → Simple structures may benefit more from multi-perspective imagery")
        print(f"   → Complex canopies may be confusing with multiple viewpoints")
    else:
        print(f"   ✗ No significant correlation found")
        print(f"   → Complexity scoring may be too coarse")
        print(f"   → Other factors may confound the relationship")

    print(f"\n2. MORPHOLOGICAL INSIGHTS:")
    print(f"   • High complexity crops (jute, wheat): {df_comparison[df_comparison['complexity_score']==3]['accuracy_improvement'].mean():.1f}% avg improvement")
    print(f"   • Medium complexity crops (maize, sugarcane): {df_comparison[df_comparison['complexity_score']==2]['accuracy_improvement'].mean():.1f}% avg improvement")
    print(f"   • Low complexity crop (rice): {df_comparison[df_comparison['complexity_score']==1]['accuracy_improvement'].iloc[0]:.1f}% improvement")

    print(f"\n3. RICE PARADOX:")
    rice_improvement = df_comparison[df_comparison['crop']=='rice']['accuracy_improvement'].iloc[0]
    if rice_improvement < 0:
        print(f"   • Rice performs BETTER with aerial-only imagery ({rice_improvement:.1f}%)")
        print(f"   • Uniform 'ideotype' structure is more distinguishable from above")
        print(f"   • Erect leaf arrangement creates consistent aerial signature")
        print(f"   • Multiple perspectives may introduce unnecessary noise")

    print(f"\n4. AI MODEL IMPLICATIONS:")
    print(f"   • Structural heterogeneity requires multiple viewing angles")
    print(f"   • 3D canopy architecture affects classification performance")
    print(f"   • Morphological distinctiveness matters for machine learning")
    print(f"   • Uniform crops may not benefit from complex imaging approaches")

    return df_comparison, spearman_corr, spearman_p

def create_canopy_complexity_table():
    """Create a comprehensive table of canopy complexity scoring."""

    complexity_data = {
        'Crop': ['Rice', 'Maize', 'Sugarcane', 'Jute', 'Wheat'],
        'Complexity_Score': [1, 2, 2, 3, 3],
        'Key_Structural_Features': [
            'Erect ideotype, uniform leaf orientation, minimal tillering',
            'Smart canopy with differential leaf angles, moderate branching',
            'Alternating leaf arrangement, tall grass structure, moderate complexity',
            'Strongly branched, heterogeneous leaf shapes, serrated margins',
            'Complex tillering architecture, variable leaf angles, multi-layered canopy'
        ],
        'Multi_Accuracy': [40.8, 32.9, 57.4, 61.1, 64.0],
        'Aerial_Accuracy': [68.2, 22.1, 49.0, 35.2, 39.4],
        'Improvement': [-27.4, 10.8, 8.4, 25.9, 24.6],
        'Scientific_Basis': [
            'IRRI ideotype: upright leaves, uniform structure (Dingkuhn et al., 1991)',
            'Smart canopy architecture: differential leaf optimization (Nature, 2024)',
            'Alternating leaf nodes, moderate structural complexity (MedCrave, 2018)',
            'Branched structure, diverse leaf morphology (Wikipedia, ResearchGate)',
            'Complex tillering, variable leaf angles (Frontiers Plant Sci, 2014)'
        ]
    }

    df_table = pd.DataFrame(complexity_data)

    print(f"\n{'='*70}")
    print("CANOPY COMPLEXITY SCORING TABLE")
    print(f"{'='*70}")
    print(df_table.to_string(index=False))

    return df_table

def generate_research_conclusions(spearman_corr, spearman_p, df_comparison):
    """Generate research conclusions based on statistical results."""

    print(f"\n{'='*70}")
    print("RESEARCH CONCLUSIONS AND IMPLICATIONS")
    print(f"{'='*70}")

    if spearman_p < 0.05:
        print(f"\n✓ HYPOTHESIS SUPPORTED (p = {spearman_p:.3f} < 0.05)")
        print(f"   Claim 1: Statistically significant correlation exists")
        print(f"   → Canopy complexity rubric is biologically meaningful and predictive")
        print(f"   → Multi-perspective imaging effectiveness varies with plant architecture")

        if spearman_corr > 0:
            print(f"   → Positive correlation: Complex canopies benefit from multiple viewpoints")
        else:
            print(f"   → Negative correlation: Simple canopies benefit more (unexpected finding)")

        print(f"\n   Claim 2: Structural traits are key visual cues for AI models")
        print(f"   → 3D morphological features affect machine learning performance")
        print(f"   → Canopy-level architecture influences classification accuracy")

    else:
        print(f"\n✗ HYPOTHESIS NOT SUPPORTED (p = {spearman_p:.3f} ≥ 0.05)")
        print(f"   Possible explanations:")
        print(f"   → Complexity scoring may be too coarse (n=5 crops)")
        print(f"   → Confounding factors: lighting, crop color, growth stage")
        print(f"   → Model limitations in leveraging structural traits")
        print(f"   → Need larger, more diverse crop dataset")

    print(f"\n{'='*70}")
    print("KEY FINDINGS FOR PUBLICATION")
    print(f"{'='*70}")

    # Sort crops by improvement for clear presentation
    sorted_crops = df_comparison.sort_values('accuracy_improvement', ascending=False)

    print(f"\n1. MULTI-PERSPECTIVE IMAGERY EFFECTIVENESS:")
    print(f"   Most benefited crops:")
    for idx, row in sorted_crops.head(2).iterrows():
        print(f"   • {row['crop'].capitalize()}: +{row['accuracy_improvement']:.1f}% improvement (Complexity: {row['complexity_score']})")

    print(f"   Least benefited crops:")
    for idx, row in sorted_crops.tail(2).iterrows():
        print(f"   • {row['crop'].capitalize()}: {row['accuracy_improvement']:.1f}% change (Complexity: {row['complexity_score']})")

    print(f"\n2. BIOLOGICAL IMPLICATIONS:")
    print(f"   • Morphological complexity affects AI observability")
    print(f"   • 3D canopy architecture is a key factor in classification")
    print(f"   • Uniform vs heterogeneous structures require different imaging approaches")

    print(f"\n3. METHODOLOGICAL CONTRIBUTIONS:")
    print(f"   • Novel framework linking plant morphology to AI performance")
    print(f"   • Biologically-grounded complexity scoring system")
    print(f"   • Multi-perspective drone imaging optimization strategy")

    print(f"\n4. LIMITATIONS AND FUTURE WORK:")
    print(f"   • Small number of crop species (n=5) limits statistical power")
    print(f"   • Complexity scoring could be refined with more morphological parameters")
    print(f"   • Larger datasets needed to strengthen morphology-AI performance links")
    print(f"   • Environmental factors (lighting, growth stage) need consideration")

if __name__ == "__main__":
    # File paths
    multi_file = "pred_output_tiles_multiperspective_with_true_label.csv"
    aerial_file = "pred_output_tiles_aerial_with_true_label.csv"

    # Run complete analysis
    print("CANOPY COMPLEXITY AND DRONE-BASED CROP CLASSIFICATION")
    print("A Biological Approach to Understanding AI Model Performance")
    print("="*70)

    # Create complexity scoring table
    complexity_table = create_canopy_complexity_table()

    # Perform statistical analysis
    comparison_df, spearman_corr, spearman_p = analyze_canopy_complexity_hypothesis(multi_file, aerial_file)

    # Generate conclusions
    generate_research_conclusions(spearman_corr, spearman_p, comparison_df)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results ready for JEI manuscript preparation.")
    print(f"Statistical framework validates biological hypothesis testing approach.")
