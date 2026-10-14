import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load and preprocess the Digits dataset
digits = load_digits()
X = digits.data  # 1797 samples, 64 features (8x8 pixel values)
y = digits.target  # Ground truth labels (0-9)

# Standardize feature values (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Experiment with different numbers of clusters (k = 2 to 16)[cite: 5]
k_range = list(range(2, 17))
inertias = []
silhouette_scores = []
ari_scores = []  # Adjusted Rand Index (measures agreement with true digit classes)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    ari_scores.append(adjusted_rand_score(y, labels))

# 3. Tabulate evaluation metrics across values of K[cite: 5]
results_df = pd.DataFrame({
    "K": k_range,
    "Inertia (WCSS)": [round(val, 2) for val in inertias],
    "Silhouette Score": [round(val, 4) for val in silhouette_scores],
    "Adjusted Rand Index": [round(val, 4) for val in ari_scores],
})
print("=" * 20 + " K-Means Clustering Performance Across K " + "=" * 20)
print(results_df.to_string(index=False))

# 4. Plot Inertia and Silhouette Score vs K[cite: 5]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Curve (Inertia)
axes[0].plot(k_range, inertias, marker="o", color="#1f77b4", linewidth=2)
axes[0].set_title("Elbow Method: Inertia vs K")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
axes[0].set_xticks(k_range)
axes[0].grid(True, linestyle="--", alpha=0.6)

# Silhouette Score Curve
axes[1].plot(k_range, silhouette_scores, marker="s", color="#ff7f0e", linewidth=2)
axes[1].set_title("Silhouette Score vs K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Mean Silhouette Coefficient")
axes[1].set_xticks(k_range)
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# 5. Fit the standard K=10 model and visualize clusters in 2D using PCA
k_optimal = 10
kmeans_10 = KMeans(n_clusters=k_optimal, init="k-means++", n_init=10, random_state=42)
labels_10 = kmeans_10.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans_10.cluster_centers_)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=labels_10, cmap="tab10", s=30, alpha=0.7, edgecolors="none"
)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="black",
    marker="X",
    s=200,
    linewidths=1.5,
    label="Cluster Centroids",
)
plt.title(f"K-Means (K=10) Projected onto First Two Principal Components")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.colorbar(scatter, label="Assigned Cluster ID")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# 6. Visualize the 10 learned cluster centroids as 8x8 digit representations
cluster_centers = scaler.inverse_transform(kmeans_10.cluster_centers_)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle("Learned Cluster Centroids (Reshaped 8x8 Digit Images)", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.imshow(cluster_centers[i].reshape(8, 8), cmap="binary", interpolation="nearest")
    ax.set_title(f"Cluster {i}")
    ax.axis("off")
plt.tight_layout()
plt.show()
