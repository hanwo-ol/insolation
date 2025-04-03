import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer # For finding optimal k
import os

# --- Configuration ---
file_path = 'inference_metrics_all_s1_t_da_exp_v2_160_v2.csv'
output_dir = 'clustering_analysis_160' # Directory to save results and plots

# --- Create Output Directory ---
os.makedirs(output_dir, exist_ok=True)
print(f"Output will be saved in: {output_dir}")

# --- Load Data ---
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
    print("DataFrame Info:")
    df.info()
    # Handle potential string representations of lists/numbers if needed
    # (Example: If DA_Level is '[1, 2]' string) - Assuming it's loaded correctly for now.
    # If DA_Level is loaded as string:
    # df['DA_Level'] = df['DA_Level'].astype(str) # Ensure consistent type first
    # If LP is loaded incorrectly:
    # df['LP'] = pd.to_numeric(df['LP'], errors='coerce') # Convert LP if needed
    # df.dropna(subset=['LP'], inplace=True) # Drop rows where LP couldn't be converted
    # df['LP'] = df['LP'].astype(int)

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error loading or initial processing of data: {e}")
    exit()

# Check if required columns exist
required_cols = ['DA_Level', 'LP', 'Filename']
if not all(col in df.columns for col in required_cols):
    print(f"Error: Missing required columns. Found: {df.columns}. Required: {required_cols}")
    exit()

# --- Feature Selection ---
orig_metrics = [col for col in df.columns if col.endswith('_Orig')]
if not orig_metrics:
    print("Error: No columns ending with '_Orig' found for clustering features.")
    exit()
print(f"\nUsing original metrics as features for clustering: {orig_metrics}")

# --- Clustering within each DA_Level and LP group ---
results_list = [] # To store dataframes with cluster labels from each group
pca_results_list = [] # To store dataframes with PCA components

# Check for NaN values in features *before* grouping
if df[orig_metrics].isnull().values.any():
    print("\nWarning: NaNs found in original metric columns. Filling with the global mean of each column.")
    for col in orig_metrics:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    print("NaNs filled.")

# Group data
grouped = df.groupby(['DA_Level', 'LP'])

for name, group in grouped:
    da_level, lp = name
    print(f"\n--- Processing Group: DA_Level={da_level}, LP={lp} ---")

    if len(group) < 2:
        print(f"Skipping group {name}: Not enough data points ({len(group)}) for clustering/analysis.")
        # Add group data without cluster/PCA if needed for completeness
        group_copy = group.copy()
        group_copy['Cluster'] = -1 # Indicate skipped group
        group_copy['PCA1'] = np.nan
        group_copy['PCA2'] = np.nan
        results_list.append(group_copy)
        pca_results_list.append(group_copy[['DA_Level', 'LP', 'Filename', 'Cluster', 'PCA1', 'PCA2'] + orig_metrics])
        continue

    # Prepare features for this group
    features = group[orig_metrics].copy() # Make a copy to avoid SettingWithCopyWarning

    # --- Preprocessing: Scale features ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --- Determine Optimal Number of Clusters (k) using Elbow Method ---
    print("Finding optimal k using Elbow method...")
    # Set a reasonable max k, avoiding k >= n_samples
    max_k = min(10, len(group) - 1)
    if max_k < 2:
        print("Not enough samples to determine optimal k > 1. Using k=1.")
        n_clusters = 1
    else:
        # Suppress FutureWarnings from KMeans if needed (older sklearn versions)
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')

        model = KMeans(random_state=42, n_init=10) # Use 'auto' or 10 for newer sklearn
        visualizer = KElbowVisualizer(model, k=(2, max_k), timings=False) # timings=False speeds it up

        try:
            visualizer.fit(scaled_features)
            n_clusters = visualizer.elbow_value_
            if n_clusters is None:
                print(f"Warning: Elbow method did not find a clear elbow for group {name}. Checking silhouette score or defaulting.")
                # Optional: Add silhouette score check here if elbow is unclear
                # For simplicity, defaulting to a common value like 3 if elbow fails
                n_clusters = min(3, max_k)
                print(f"Defaulting to k={n_clusters} for group {name}.")
            else:
                print(f"Optimal k determined: {n_clusters}")

            # Save the elbow plot
            elbow_plot_filename = os.path.join(output_dir, f'elbow_plot_DA{da_level}_LP{lp}.png')
            visualizer.show(outpath=elbow_plot_filename, clear_figure=True)
            print(f"Saved elbow plot to {elbow_plot_filename}")
            # plt.close(visualizer.figure_) # visualizer.show should handle closing
        except Exception as e:
            print(f"Error during KElbowVisualizer for group {name}: {e}. Defaulting to k=3 (if possible).")
            n_clusters = min(3, max_k) if max_k >= 1 else 1
            # plt.close() # Ensure figure is closed if error occurs

        # Reset warnings
        warnings.filterwarnings("default", category=FutureWarning, module='sklearn.cluster._kmeans')


    # Handle case where only 1 cluster is possible/chosen
    if n_clusters < 1:
         print(f"Cannot perform clustering with k={n_clusters}. Setting k=1.")
         n_clusters = 1

    # --- Perform K-Means Clustering ---
    print(f"Performing K-Means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # --- Assign Cluster Labels ---
    group_with_clusters = group.copy()
    group_with_clusters['Cluster'] = cluster_labels
    results_list.append(group_with_clusters) # Add to the main list

    # --- Dimensionality Reduction for Visualization (PCA) ---
    if n_clusters > 1 and len(group) >= 2: # Need at least 2 points and >1 cluster for meaningful PCA plot
        print("Performing PCA for visualization...")
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(scaled_features)

        # Create DataFrame for PCA results
        pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'], index=group.index)
        group_pca_vis = pd.concat([group_with_clusters[['DA_Level', 'LP', 'Filename', 'Cluster'] + orig_metrics], pca_df], axis=1)
        pca_results_list.append(group_pca_vis) # Add PCA results

        # --- Visualize Clusters ---
        plt.figure(figsize=(10, 7))
        palette = sns.color_palette('viridis', n_colors=n_clusters)
        sns.scatterplot(data=group_pca_vis, x='PCA1', y='PCA2', hue='Cluster', palette=palette, s=50, alpha=0.8)
        plt.title(f'Clusters for DA_Level={da_level}, LP={lp} (PCA Visualization)')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.legend(title='Cluster')
        plt.grid(True, linestyle='--', alpha=0.6)

        pca_plot_filename = os.path.join(output_dir, f'pca_clusters_DA{da_level}_LP{lp}.png')
        plt.savefig(pca_plot_filename, bbox_inches='tight')
        print(f"Saved PCA cluster plot to {pca_plot_filename}")
        plt.close()
    else:
        print("Skipping PCA visualization (only 1 cluster or insufficient data).")
        # Add group data with NaN PCA components if needed for consistent structure
        group_pca_vis = group_with_clusters.copy()
        group_pca_vis['PCA1'] = np.nan
        group_pca_vis['PCA2'] = np.nan
        pca_results_list.append(group_pca_vis[['DA_Level', 'LP', 'Filename', 'Cluster', 'PCA1', 'PCA2'] + orig_metrics])


# --- Combine Results ---
if not results_list:
    print("\nNo groups were successfully processed. No combined results generated.")
else:
    df_clustered = pd.concat(results_list, ignore_index=True)
    print("\n--- Combined DataFrame with Cluster Assignments (First 10 rows) ---")
    print(df_clustered.head(10))

    # Save the full dataframe with cluster assignments
    clustered_file = os.path.join(output_dir, 'data_with_clusters.csv')
    df_clustered.to_csv(clustered_file, index=False)
    print(f"\nFull data with cluster assignments saved to: {clustered_file}")

    # --- Analyze Cluster Characteristics ---
    print("\n--- Cluster Analysis (Mean Metrics per Cluster/Group) ---")
    # Calculate mean of original metrics for each cluster within each group
    cluster_analysis = df_clustered.groupby(['DA_Level', 'LP', 'Cluster'])[orig_metrics].mean()
    # Optionally add cluster sizes
    cluster_analysis['Size'] = df_clustered.groupby(['DA_Level', 'LP', 'Cluster']).size()
    print(cluster_analysis)

    # Save cluster analysis
    analysis_file = os.path.join(output_dir, 'cluster_analysis_means.csv')
    cluster_analysis.to_csv(analysis_file)
    print(f"\nCluster analysis (mean metrics per cluster) saved to: {analysis_file}")

    # --- Save PCA results (if generated) ---
    if pca_results_list:
         df_pca_clustered = pd.concat(pca_results_list, ignore_index=True)
         pca_file = os.path.join(output_dir, 'data_with_pca_and_clusters.csv')
         df_pca_clustered.to_csv(pca_file, index=False)
         print(f"Data including PCA components saved to: {pca_file}")

    # --- Example: Identifying Files in Specific Clusters (e.g., lowest PSNR cluster) ---
    print("\n--- Example: Identifying Files in Clusters with Lowest Mean PSNR_Orig ---")
    if 'PSNR_Orig' in cluster_analysis.columns:
        lowest_psnr_clusters = cluster_analysis.loc[cluster_analysis.groupby(['DA_Level', 'LP'])['PSNR_Orig'].idxmin()]
        print("\nClusters with the lowest average PSNR_Orig for each DA_Level/LP group:")
        print(lowest_psnr_clusters)

        print("\nExample Filenames from lowest PSNR_Orig clusters:")
        for index, row in lowest_psnr_clusters.iterrows():
            da, lp, cluster = index # Get group and cluster index
            print(f"\nGroup DA={da}, LP={lp}, Lowest PSNR Cluster={cluster}:")
            filenames = df_clustered[
                (df_clustered['DA_Level'] == da) &
                (df_clustered['LP'] == lp) &
                (df_clustered['Cluster'] == cluster)
            ]['Filename'].head(5).tolist() # Get first 5 filenames
            print(filenames)
    else:
        print("PSNR_Orig column not found in analysis, skipping lowest PSNR example.")

print("\n--- Clustering and Analysis Complete ---")
