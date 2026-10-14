import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load and preprocess the Mall Customers dataset

df = pd.read_excel("Mall_customers/Mall_Customers.xlsx")

# Standardize column names to handle variations (e.g., 'Genre' vs 'Gender')
df.columns = df.columns.str.strip()

# Select the two primary features commonly analyzed for customer segmentation
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Optimal cluster determination (Elbow method & Dendrogram)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Elbow Curve for K-means
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_tmp = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans_tmp.fit(X_scaled)
    wcss.append(kmeans_tmp.inertia_)

axes[0].plot(k_range, wcss, marker="o", color="royalblue")
axes[0].set_title("K-Means Elbow Method (Inertia vs K)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].grid(True)

# Dendrogram for Agglomerative Clustering
linked = linkage(X_scaled, method="ward")
dendrogram(linked, ax=axes[1], truncate_mode="lastp", p=30, show_leaf_counts=True)
axes[1].set_title("Hierarchical Dendrogram (Ward Linkage)")
axes[1].set_xlabel("Sample Index / Cluster Size")
axes[1].set_ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

# Both the elbow point and dendrogram indicate k = 5 optimal clusters
optimal_k = 5

# 3. Fit K-Means and Agglomerative Clustering
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, metric="euclidean", linkage="ward")
agg_labels = agg_clustering.fit_predict(X_scaled)

# 4. Compute Evaluation Metrics
# For Agglomerative clustering, inertia is computed manually via distance to empirical centroids
agg_centroids = np.array([X_scaled[agg_labels == i].mean(axis=0) for i in range(optimal_k)])
agg_inertia = sum(np.sum((X_scaled[agg_labels == i] - agg_centroids[i]) ** 2) for i in range(optimal_k))

metrics = {
    "Metric": ["Inertia (WCSS)", "Silhouette Score", "Davies-Bouldin Index"],
    "K-Means": [
        round(kmeans.inertia_, 2),
        round(silhouette_score(X_scaled, kmeans_labels), 4),
        round(davies_bouldin_score(X_scaled, kmeans_labels), 4),
    ],
    "Hierarchical (Ward)": [
        round(agg_inertia, 2),
        round(silhouette_score(X_scaled, agg_labels), 4),
        round(davies_bouldin_score(X_scaled, agg_labels), 4),
    ],
}

print("=" * 20 + " Clustering Metrics Comparison " + "=" * 20)
print(pd.DataFrame(metrics).to_string(index=False))

# 5. Visualizing and comparing cluster assignments[cite: 4]
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Scatter plot for K-Means
scatter_km = axes[0].scatter(
    X[:, 0], X[:, 1], c=kmeans_labels, cmap="viridis", s=60, edgecolors="k", alpha=0.85
)
# Inverse transform centroids to plot on raw scale
km_centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
axes[0].scatter(
    km_centers_unscaled[:, 0],
    km_centers_unscaled[:, 1],
    c="red",
    s=220,
    marker="X",
    label="Centroids",
)
axes[0].set_title(f"K-Means Clustering (k={optimal_k})")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score (1-100)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Scatter plot for Agglomerative Clustering
scatter_agg = axes[1].scatter(
    X[:, 0], X[:, 1], c=agg_labels, cmap="viridis", s=60, edgecolors="k", alpha=0.85
)
agg_centers_unscaled = scaler.inverse_transform(agg_centroids)
axes[1].scatter(
    agg_centers_unscaled[:, 0],
    agg_centers_unscaled[:, 1],
    c="red",
    s=220,
    marker="X",
    label="Centroids",
)
axes[1].set_title(f"Agglomerative Clustering (k={optimal_k})")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
