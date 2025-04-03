import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re 

# --- Configuration ---
# Directory where your CSV files are located (leave empty if in the same directory)
data_directory = ""
# Pattern to match your CSV files
# Assumes files are named like: train_metrics_s1_t_da_exp_v2_lp0_da{LEVEL}_e160_b4_lr0.0012.csv
file_pattern = "train_metrics_s1_t_da_exp_v2_lp0_da*_e160_b4_lr0.0012.csv"
# Number of last epochs to consider for "final" performance statistics
last_n_epochs = 40

# --- Find Files ---
search_path = os.path.join(data_directory, file_pattern)
csv_files = glob.glob(search_path)

if not csv_files:
    print(f"Error: No files found matching pattern '{search_path}'")
    exit()

print(f"Found {len(csv_files)} files:")
for f in csv_files:
    print(f"- {os.path.basename(f)}")

# --- Load and Combine Data ---
all_data = []
da_levels_found = []

for file in csv_files:
    # Extract DA level from filename
    filename = os.path.basename(file)
    # Try to extract the part between '_da' and '_e160'
    match = re.search(r'_da(.*?)_e160', filename)
    if match:
        da_level_str = match.group(1)
        da_label = f"DA_{da_level_str}" # e.g., DA_0, DA_1, DA_[1, 2]
        da_levels_found.append(da_label)
    else:
        print(f"Warning: Could not extract DA level from filename '{filename}'. Skipping.")
        continue

    try:
        df = pd.read_csv(file)
        # Add DA level identifier and original filename for reference
        df['DA_Level'] = da_label
        df['Source_File'] = filename
        # Ensure 'Epoch' column exists and is numeric (sometimes needed if header issues)
        if 'Epoch' not in df.columns:
             print(f"Warning: 'Epoch' column not found in {filename}. Attempting to use index + 1.")
             df['Epoch'] = df.index + 1
        else:
            df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce') # Ensure numeric

        df = df.dropna(subset=['Epoch']) # Drop rows where Epoch couldn't be parsed

        all_data.append(df)
    except Exception as e:
        print(f"Error reading or processing file {filename}: {e}")

if not all_data:
    print("Error: No data could be loaded successfully.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Ensure metric columns are numeric
metric_cols = ['Loss', 'MSE', 'MAE', 'PSNR', 'SSIM']
for col in metric_cols:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    else:
        print(f"Warning: Metric column '{col}' not found in combined data.")
        metric_cols.remove(col) # Remove if not present

# Drop rows where essential data might be missing after conversion
combined_df = combined_df.dropna(subset=['Epoch', 'DA_Level'] + metric_cols)

print(f"\n--- Data Loading Complete ({len(combined_df)} total rows) ---")
print("Unique DA Levels identified:", combined_df['DA_Level'].unique())

# --- Calculate Statistics ---

# 1. Statistics over ALL epochs
print(f"\n--- Statistics Over All Epochs (1 to {combined_df['Epoch'].max()}) ---")
stats_all = combined_df.groupby('DA_Level')[metric_cols].agg(['mean', 'std'])
print(stats_all)

# 2. Statistics over the LAST N epochs
print(f"\n--- Statistics Over Last {last_n_epochs} Epochs ---")
max_epoch = combined_df['Epoch'].max()
stats_last_n = combined_df[combined_df['Epoch'] > (max_epoch - last_n_epochs)] \
                  .groupby('DA_Level')[metric_cols] \
                  .agg(['mean', 'std'])
print(stats_last_n)

# --- Visualize Metrics ---
print("\n--- Generating Plots ---")

num_metrics = len(metric_cols)
# Adjust layout based on number of metrics
num_cols = 1 if num_metrics > 1 else 1
num_rows = (num_metrics + num_cols - 1) // num_cols

plt.style.use('seaborn-v0_8-whitegrid') # Use a clean seaborn style
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 5 * num_rows), squeeze=False)
axes = axes.flatten() # Flatten the axes array for easy iteration

# Use seaborn's default color palette or define your own
palette = sns.color_palette("tab10", n_colors=len(combined_df['DA_Level'].unique()))

for i, metric in enumerate(metric_cols):
    ax = axes[i]
    sns.lineplot(
        data=combined_df,
        x='Epoch',
        y=metric,
        hue='DA_Level',
        palette=palette,
        errorbar=None, # Plot mean line without error bars for clarity initially
        # errorbar='sd', # Uncomment this to show standard deviation bands
        ax=ax,
        linewidth=1.5
    )
    ax.set_title(f'{metric} vs. Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend(title='DA Level')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add markers for the last epoch value for each DA level for quick comparison
    # last_epoch_data = combined_df[combined_df['Epoch'] == max_epoch]
    # sns.scatterplot(data=last_epoch_data, x='Epoch', y=metric, hue='DA_Level', ax=ax, legend=False, s=50, marker='o')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
fig.suptitle('Training Metrics Comparison Across DA Levels Setting 1', fontsize=16, y=0.99)
plt.show()

print("\n--- Analysis Complete ---")
