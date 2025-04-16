import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


# Load the ARFF dataset
def load_arff_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    # Convert any bytes columns to string
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    return df


# Function to calculate Manhattan distance
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))


# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Implementation of K-medoids clustering algorithm
class KMedoids:
    def __init__(self, n_clusters=3, max_iterations=100, distance_metric='euclidean', random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.medoids_indices = None
        self.labels = None
        self.distance_metric = distance_metric

        if distance_metric == 'manhattan':
            self.distance_function = manhattan_distance
        else:  # default to euclidean
            self.distance_function = euclidean_distance

    def _init_medoids(self, X):
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Randomly select initial medoids
        n_samples = X.shape[0]
        self.medoids_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[self.medoids_indices]

    def _assign_points_to_clusters(self, X):
        # Calculate distances from each point to each medoid
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i in range(self.n_clusters):
            for j in range(X.shape[0]):
                distances[j, i] = self.distance_function(X[j], self.medoids[i])

        # Assign each point to the nearest medoid
        self.labels = np.argmin(distances, axis=1)

        return distances

    def _update_medoids(self, X, distances):
        # For each cluster, select the point that minimizes the sum of distances to other points in the cluster
        for i in range(self.n_clusters):
            cluster_points_indices = np.where(self.labels == i)[0]

            if len(cluster_points_indices) == 0:
                # If a cluster is empty, reinitialize its medoid randomly
                self.medoids_indices[i] = np.random.choice(X.shape[0])
                self.medoids[i] = X[self.medoids_indices[i]]
                continue

            # Calculate the cost (sum of distances) for each point in the cluster
            cluster_costs = np.zeros(len(cluster_points_indices))

            for j, point_idx in enumerate(cluster_points_indices):
                # Sum of distances from this point to all other points in the cluster
                cluster_costs[j] = sum(self.distance_function(X[point_idx], X[other_idx])
                                       for other_idx in cluster_points_indices)

            # Select the point with the minimum cost as the new medoid
            min_cost_idx = np.argmin(cluster_costs)
            new_medoid_idx = cluster_points_indices[min_cost_idx]

            self.medoids_indices[i] = new_medoid_idx
            self.medoids[i] = X[new_medoid_idx]

    def _calculate_total_cost(self, X, distances):
        # Sum of distances from each point to its assigned medoid
        return sum(distances[i, self.labels[i]] for i in range(X.shape[0]))

    def fit(self, X):
        self._init_medoids(X)

        prev_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Assign points to clusters
            distances = self._assign_points_to_clusters(X)

            # Calculate current cost
            current_cost = self._calculate_total_cost(X, distances)

            # Check for convergence
            if prev_cost - current_cost < 1e-4:
                break

            prev_cost = current_cost

            # Update medoids
            self._update_medoids(X, distances)

        return self

    def predict(self, X):
        # Calculate distances from each point to each medoid
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i in range(self.n_clusters):
            for j in range(X.shape[0]):
                distances[j, i] = self.distance_function(X[j], self.medoids[i])

        # Assign each point to the nearest medoid
        return np.argmin(distances, axis=1)


# Main execution
if __name__ == "__main__":
    # Load and prepare the dataset
    file_path = "student_Depression_dataset_pre.csv.arff"
    print(f"Loading dataset from {file_path}...")
    df = load_arff_data(file_path)
    
    # Preprocess the data
    print("Preprocessing data...")
    
    # For Sleep Duration column, map values to numerical categories
    if 'Sleep Duration' in df.columns:
        sleep_duration_mapping = {
            'Less than 5 hours': 0,
            '5-6 hours': 1,
            '7-8 hours': 2,
            'More than 8 hours': 3,
            'Others': 4
        }
        df['Sleep Duration'] = df['Sleep Duration'].map(sleep_duration_mapping)
        print("Mapped Sleep Duration values to numerical categories")
    else:
        print("Warning: 'Sleep Duration' column not found in the dataset.")
    
    # For Dietary Habits column, map values to numerical categories
    if 'Dietary Habits' in df.columns:
        dietary_habits_mapping = {
            'Unhealthy': 0,
            'Moderate': 1,
            'Healthy': 2,
            'Others': 3
        }
        df['Dietary Habits'] = df['Dietary Habits'].map(dietary_habits_mapping)
        print("Mapped Dietary Habits values to numerical categories")
    else:
        print("Warning: 'Dietary Habits' column not found in the dataset.")

    # Display basic dataset information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Exclude 'Depression' column for clustering but keep it in the original dataframe
    if 'Depression' in df.columns:
        print("Excluding 'Depression' column from clustering features...")
        X_features = df.drop(columns=['Depression'])
    else:
        print("Warning: 'Depression' column not found in the dataset.")
        X_features = df.copy()

    # Randomly select 1000 samples from the dataset
    sample_size = 500
    if len(df) > sample_size:
        print(f"Randomly selecting {sample_size} samples from the dataset...")
        # Set random seed for reproducibility
        np.random.seed(42)
        sampled_indices = np.random.choice(len(df), sample_size, replace=False)
        df_sampled = df.iloc[sampled_indices].reset_index(drop=True)
        # Also sample the features dataframe using the same indices
        X_features_sampled = X_features.iloc[sampled_indices].reset_index(drop=True)
        print(f"Sampled dataset shape: {df_sampled.shape}")
    else:
        print(f"Dataset has fewer than {sample_size} samples. Using the entire dataset.")
        df_sampled = df
        X_features_sampled = X_features

    # Prepare the data for clustering - use only the features, not the Depression column
    X = X_features_sampled.values

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction to 2 dimensions
    print("Applying PCA to reduce data to 2 dimensions...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    # Determine the optimal number of clusters using various evaluation metrics
    max_clusters = min(10, X_pca.shape[0] - 1)  # Set a reasonable upper limit
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    k_values = range(2, max_clusters + 1)

    print("Calculating clustering evaluation metrics for different K values...")

    for k in k_values:
        kmedoids = KMedoids(n_clusters=k, max_iterations=100, random_state=42)
        kmedoids.fit(X_pca)
        labels = kmedoids.labels

        # Calculate evaluation metrics
        try:
            # Silhouette Score (higher is better)
            sil_score = silhouette_score(X_pca, labels)
            silhouette_scores.append(sil_score)
            
            # Calinski-Harabasz Index (higher is better)
            ch_score = calinski_harabasz_score(X_pca, labels)
            calinski_scores.append(ch_score)
            
            # Davies-Bouldin Index (lower is better)
            db_score = davies_bouldin_score(X_pca, labels)
            davies_bouldin_scores.append(db_score)
            
            print(f"K = {k}, Silhouette = {sil_score:.4f}, Calinski-Harabasz = {ch_score:.2f}, Davies-Bouldin = {db_score:.4f}")
        except Exception as e:
            print(f"Error calculating metrics for K={k}: {e}")
            silhouette_scores.append(-1)
            calinski_scores.append(-1)
            davies_bouldin_scores.append(float('inf'))

    # Find the optimal K based on Silhouette Score
    optimal_k_sil = k_values[np.argmax(silhouette_scores)]
    # For Calinski-Harabasz, higher is better
    optimal_k_ch = k_values[np.argmax(calinski_scores)]
    # For Davies-Bouldin, lower is better
    optimal_k_db = k_values[np.argmin(davies_bouldin_scores)]
    
    print(f"\nOptimal K based on Silhouette Score = {optimal_k_sil} (Score: {max(silhouette_scores):.4f})")
    print(f"Optimal K based on Calinski-Harabasz Index = {optimal_k_ch} (Score: {max(calinski_scores):.2f})")
    print(f"Optimal K based on Davies-Bouldin Index = {optimal_k_db} (Score: {min(davies_bouldin_scores):.4f})")
    
    # Use the Silhouette Score's optimal K as our final choice
    optimal_k = optimal_k_sil
    best_silhouette = max(silhouette_scores)
    best_calinski = max(calinski_scores)
    best_davies = min(davies_bouldin_scores)

    # Original Silhouette Score plot for backward compatibility
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'o-', color='blue')
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different K values')
    plt.legend()
    plt.grid(True)
    plt.savefig('silhouette_scores.png')

    # Apply K-medoids with optimal K
    print(f"\nApplying K-medoids with optimal K = {optimal_k}...")
    start_time = time.time()
    final_kmedoids = KMedoids(n_clusters=optimal_k, max_iterations=100, random_state=42)
    final_kmedoids.fit(X_pca)
    end_time = time.time()

    print(f"K-medoids clustering completed in {end_time - start_time:.2f} seconds")

    # Get the cluster assignments
    cluster_labels = final_kmedoids.labels

    # Add the cluster labels to the original dataframe (which still contains the Depression column)
    df_sampled['cluster'] = cluster_labels

    # Count samples in each cluster
    cluster_counts = df_sampled['cluster'].value_counts().sort_index()
    print("\nNumber of samples in each cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} samples")

    # Save results to csv
    df_sampled.to_csv('clustering_results_500.csv', index=False)
    print("Results saved to 'clustering_results.csv'")

    # Visualize the clusters in 2D space (PCA)
    plt.figure(figsize=(12, 10))
    for i in range(optimal_k):
        cluster_points = X_pca[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    
    # Plot medoids
    plt.scatter(final_kmedoids.medoids[:, 0], final_kmedoids.medoids[:, 1], 
                s=200, c='red', marker='*', label='Medoids')
    
    # Add metrics information to the title and figure
    plt.title(f'K-medoids Clustering after PCA (K={optimal_k})', fontsize=14)
    
    # Add metrics information as text on the figure
    plt.figtext(0.5, 0.01, 
                f"Metrics for K={optimal_k}:\n" +
                f"Silhouette Score: {best_silhouette:.4f} (higher is better)\n" +
                f"Calinski-Harabasz Index: {best_calinski:.2f} (higher is better)\n" +
                f"Davies-Bouldin Index: {best_davies:.4f} (lower is better)", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Adjust layout to make room for metrics text
    plt.savefig('kmedoids_clustering_pca_500.png', bbox_inches='tight')
    print("PCA clustering visualization with metrics saved to 'kmedoids_clustering_pca.png'")