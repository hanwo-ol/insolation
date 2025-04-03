import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
file_path = 'inference_metrics_all_s1_t_da_exp_v2_160_v2.csv'
output_dir = 'inference_metrics_all_s1_t_da_exp_v2_160_v2' # Directory to save plots

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
    print("First 5 rows:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print("\nUnique DA_Level values:", df['DA_Level'].unique())
    print("Unique LP values:", df['LP'].unique())
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Identify Metric Columns ---
metric_columns = [col for col in df.columns if col not in ['DA_Level', 'LP', 'Filename']]
base_metrics = ['MSE', 'MAE', 'PSNR', 'SSIM']
processing_types = ['Orig', 'Clip', 'MinMax']

print(f"\nIdentified Metric Columns: {metric_columns}")

# --- 1. Descriptive Statistics per DA_Level and LP ---
print("\n--- Calculating Descriptive Statistics (Mean and Variance) ---")
grouped_data = df.groupby(['DA_Level', 'LP'])
stats = grouped_data[metric_columns].agg(['mean', 'var'])

print("\nDescriptive Statistics (Mean and Variance) per DA_Level and LP:")
print(stats)

# Save stats to CSV
stats_file = os.path.join(output_dir, 'descriptive_stats.csv')
stats.to_csv(stats_file)
print(f"\nDescriptive statistics saved to {stats_file}")


# --- 2. Histograms for Each Metric per Group ---
print("\n--- Generating Histograms for Metric Distributions ---")

for metric in metric_columns:
    plt.figure(figsize=(12, 8))
    plot_title = f'Distribution of {metric}'
    print(f"Plotting: {plot_title}")

    # Determine common reasonable limits, especially for error metrics (MSE, MAE) vs quality metrics (PSNR, SSIM)
    # This helps in comparing distributions visually
    # Calculate quantiles to avoid extreme outliers skewing the view
    q_low = df[metric].quantile(0.01)
    q_high = df[metric].quantile(0.99)
    if 'PSNR' in metric: # PSNR often has a wider range, adjust if needed based on data
         q_low = max(0, df[metric].quantile(0.01) - 5) # Allow slightly lower view
         q_high = df[metric].quantile(0.99) + 5  # Allow slightly higher view
    elif 'SSIM' in metric:
         q_low = max(0, df[metric].quantile(0.01) - 0.1)
         q_high = min(1, df[metric].quantile(0.99) + 0.1)
    else: # MSE, MAE
        q_low = max(0, q_low) # Error can't be negative
        q_high = q_high * 1.2 # Allow some headroom

    bin_range = (q_low, q_high)

    # Use seaborn's FacetGrid for cleaner multi-group histograms
    try:
        g = sns.FacetGrid(df, col="LP", row="DA_Level", hue="LP", # Hue by LP within facets
                          sharex=False, sharey=False, height=3, aspect=1.5,
                          margin_titles=True)
        g.map(sns.histplot, metric, kde=True, stat="density", common_norm=False, bins=15, binrange=bin_range)
        g.add_legend(title='LP')
        g.fig.suptitle(plot_title, y=1.03) # Add title above facets
        g.set_axis_labels(metric, "Density")
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for suptitle

        # Save the plot
        plot_filename = os.path.join(output_dir, f'hist_{metric}.png')
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Saved histogram to {plot_filename}")
        plt.close() # Close the figure to free memory
    except Exception as e:
        print(f"Error plotting histogram for {metric}: {e}")
        plt.close() # Ensure plot is closed even if error occurs


# --- 3. Compare Orig, Clip, MinMax within each DA_Level/LP group ---
print("\n--- Generating Box Plots to Compare Processing Types (Orig, Clip, MinMax) ---")

# Use melt to reshape the data for easier plotting with seaborn
id_vars = ['DA_Level', 'LP']
df_melted_all = pd.DataFrame() # To store all melted data

for base in base_metrics:
    cols_for_metric = [f"{base}_{ptype}" for ptype in processing_types]
    # Ensure all expected columns exist
    if all(col in df.columns for col in cols_for_metric):
        temp_melted = df.melt(id_vars=id_vars,
                              value_vars=cols_for_metric,
                              var_name='Metric_Type',
                              value_name='Value')
        temp_melted['Base_Metric'] = base
        temp_melted['Processing'] = temp_melted['Metric_Type'].str.split('_').str[1]
        df_melted_all = pd.concat([df_melted_all, temp_melted], ignore_index=True)
    else:
        print(f"Warning: Skipping {base} for melted plot - missing one or more columns: {cols_for_metric}")

if not df_melted_all.empty:
    # Create faceted box plots comparing processing types for each base metric
    print("Plotting: Box plots comparing Orig, Clip, MinMax")
    try:
        g = sns.catplot(
            data=df_melted_all,
            x='Processing',
            y='Value',
            col='LP',
            row='DA_Level',
            hue='Base_Metric', # Color by base metric within each facet
            kind='box',
            order=processing_types, # Ensure consistent order
            sharey=False, # Scales can differ significantly (MSE vs PSNR)
            height=3, aspect=1.5,
            margin_titles=True,
            legend=False # Add legend manually later if needed
        )
        g.set_axis_labels("Processing Type", "Metric Value")
        g.set_titles(col_template="LP: {col_name}", row_template="DA: {row_name}")
        g.fig.suptitle('Comparison of Processing Types (Orig, Clip, MinMax) per Group', y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plot_filename = os.path.join(output_dir, 'boxplot_processing_comparison.png')
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Saved processing comparison box plot to {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error plotting processing comparison: {e}")
        plt.close()
else:
    print("Skipping processing comparison plot as no data was successfully melted.")


# --- 4. Compare Metrics Across DA_Level/LP Groups ---
print("\n--- Generating Box Plots to Compare Metrics Across Groups ---")

# Create a combined group identifier for easier plotting
df['Group'] = 'DA:' + df['DA_Level'].astype(str) + '_LP:' + df['LP'].astype(str)
group_order = sorted(df['Group'].unique(), key=lambda x: (x.split('_LP:')[0], int(x.split('_LP:')[1])))


for metric in metric_columns:
    plt.figure(figsize=(15, 7)) # Wider figure for potentially many groups
    plot_title = f'Comparison of {metric} Across DA_Level and LP Groups'
    print(f"Plotting: {plot_title}")

    try:
        sns.boxplot(data=df, x='Group', y=metric, order=group_order)
        plt.title(plot_title)
        plt.xlabel('Group (DA_Level_LP)')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f'boxplot_group_comparison_{metric}.png')
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Saved group comparison box plot for {metric} to {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error plotting group comparison for {metric}: {e}")
        plt.close()

print("\n--- Analysis Complete ---")
print(f"Plots saved in directory: {output_dir}")
