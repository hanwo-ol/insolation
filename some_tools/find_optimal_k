import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# --- Configuration ---
file_path = 'inference_metrics_all_s1_t_da_exp_v2_160_v2.csv'
output_dir = 'optimal_k_analysis_160_v2' # Directory to save plots
feature_suffix = '_Orig' # Features to use for clustering
k_range = range(2, 11) # Range of cluster numbers (k) to test (Starts from 2 for silhouette)
random_state = 42 # for reproducibility

# --- Create output directory ---
os.makedirs(output_dir, exist_ok=True)

# --- Suppress KMeans FutureWarnings for n_init ---
# KMeans behavior for n_init is changing, this suppresses the warning.
# You might need to adjust n_init explicitly in future scikit-learn versions.
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')


# --- Load Data ---
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Identify Features for Clustering ---
orig_metric_columns = [col for col in df.columns if col.endswith(feature_suffix)]
if not orig_metric_columns:
    print(f"Error: No columns found with suffix '{feature_suffix}'")
    exit()
print(f"\nUsing features for determining optimal k: {orig_metric_columns}")

# --- Iterate through each group and find optimal k ---
grouped = df.groupby(['DA_Level', 'LP'])
optimal_k_results = {} # Store suggested k for each group

print("\n--- Evaluating Optimal Number of Clusters (k) for each Group ---")

for name, group_df in grouped:
    da_level, lp = name
    group_id_str = f"DA_{da_level}_LP_{lp}"
    print(f"\nProcessing group: {group_id_str}")

    # Select features and handle potential missing values for this group
    features = group_df[orig_metric_columns].copy()
    features.dropna(inplace=True)

    # Check if group has enough samples for the k range
    if len(features) <= max(k_range):
         print(f"Skipping group {group_id_str}: Not enough valid samples ({len(features)}) for k up to {max(k_range)}.")
         k_range_group = range(2, len(features)) # Adjust k_range if too few samples
         if len(k_range_group) == 0:
             continue # Skip entirely if cannot even test k=2
    else:
        k_range_group = k_range


    if len(features) < 2: # Need at least 2 samples for silhouette/elbow
        print(f"Skipping group {group_id_str}: Less than 2 valid samples ({len(features)}).")
        continue

    print(f"  Group size after NaN drop: {len(features)}, Testing k in {list(k_range_group)}")

    # Scale features for this group
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    inertia_values = []
    silhouette_values = []

    for k in k_range_group:
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(features_scaled)

        # Calculate Inertia (for Elbow Method)
        inertia_values.append(kmeans.inertia_)

        # Calculate Silhouette Score
        try:
            score = silhouette_score(features_scaled, kmeans.labels_)
            silhouette_values.append(score)
        except ValueError:
            print(f"    Warning: Could not calculate silhouette score for k={k} (likely degenerate clustering). Appending NaN.")
            silhouette_values.append(np.nan) # Assign NaN if score calculation fails

    # --- Plotting for the current group ---

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Optimal k Evaluation for {group_id_str}', fontsize=16)

    # Elbow Method Plot
    axes[0].plot(list(k_range_group), inertia_values, marker='o', linestyle='-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (Sum of Squared Distances)')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Silhouette Score Plot
    axes[1].plot(list(k_range_group), silhouette_values, marker='o', linestyle='-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score Method')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Find optimal k based on silhouette score (if possible)
    optimal_k_silhouette = np.nan # Default to NaN
    if len(silhouette_values) > 0 and not np.all(np.isnan(silhouette_values)):
        best_k_idx = np.nanargmax(silhouette_values)
        optimal_k_silhouette = list(k_range_group)[best_k_idx]
        axes[1].axvline(optimal_k_silhouette, color='r', linestyle='--', label=f'Optimal k = {optimal_k_silhouette}')
        axes[1].legend()
        print(f"  Suggested optimal k (Max Silhouette): {optimal_k_silhouette}")
        optimal_k_results[group_id_str] = optimal_k_silhouette
    else:
        print("  Could not determine optimal k from silhouette scores.")
        optimal_k_results[group_id_str] = None


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    # Save plot
    plot_filename = os.path.join(output_dir, f'optimal_k_eval_{group_id_str}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"  Saved evaluation plot to {plot_filename}")
    plt.close()

print("\n--- Optimal k Evaluation Complete ---")
print("Suggested optimal k values based on Silhouette Score:")
for group, k in optimal_k_results.items():
    print(f"  {group}: {k if k is not None else 'Could not determine'}")

print(f"\nPlots saved in directory: {output_dir}")
