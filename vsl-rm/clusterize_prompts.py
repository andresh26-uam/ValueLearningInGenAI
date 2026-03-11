import json
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar

def clusterize():

	# Load ultrafeedback dataset
	print("Loading ultrafeedback dataset...")
	dataset = load_dataset("openbmb/UltraFeedback", split="train")

	# Extract prompts
	prompts = [item["instruction"] for item in dataset]
	sources = [item.get("source", "unknown") for item in dataset]

	print(f"Loaded {len(prompts)} prompts")

	print("Loading SentenceTransformer model...")
	model = SentenceTransformer('all-MiniLM-L6-v2', model_kwargs={"torch_dtype": "float16"})  # Use GPU if available

	print("Generating embeddings...")
	prompt_vectors = model.encode(prompts, show_progress_bar=True)

	# Apply PCA for dimensionality reduction before t-SNE
	#print("Applying PCA...")
	#pca = PCA(n_components=50)
	prompt_vectors_pca = prompt_vectors

	# K-means clustering
	n_clusters = 8
	print(f"Clustering prompts into {n_clusters} clusters...")
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	cluster_ids = kmeans.fit_predict(prompt_vectors_pca)


	# Add cluster IDs to dataset
	def add_cluster_info(item, index):
		item["cluster_id"] = int(cluster_ids[index])
		item["prompt_vector_pca"] = prompt_vectors_pca[index].tolist()  # Store PCA-transformed embedding as list for JSON serialization
		return item
	
	dataset = dataset.map(add_cluster_info, with_indices=True)
	print(dataset[0].keys())
	input("???")
	# Save dataset with cluster IDs
	print("Saving dataset with cluster IDs...")
	dataset.save_to_disk("/home/ubuntu/ValueLearningInGenAI/vsl-rm/ultrafeedback_clustered")


clusterize()
# Load the dataset back to verify
print("Loading clustered dataset to verify...")
clustered_dataset: Dataset | DatasetDict = load_from_disk("/home/ubuntu/ValueLearningInGenAI/vsl-rm/ultrafeedback_clustered")

prompt_vectors_pca = np.array([item["prompt_vector_pca"] for item in clustered_dataset]) # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
print(f"Loaded clustered dataset with {len(clustered_dataset)} samples")
#print(f"Sample cluster IDs: {[item['cluster_id'] for item in clustered_dataset[:5]]}")
# Apply t-SNE
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
prompt_2d = tsne.fit_transform(prompt_vectors_pca)
sources = [item.get("source", "unknown") for item in clustered_dataset] # pyright: ignore[reportAttributeAccessIssue]
# Define unique sources and colors
unique_sources = list(set(sources))
source_colors = {src: i for i, src in enumerate(unique_sources)}

# Plot 1: Colored by cluster ID
print("Plotting t-SNE visualization...")
plt.figure(figsize=(12, 8))
cluster_ids: list[int] = [item["cluster_id"] for item in clustered_dataset] # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType, reportAssignmentType]
scatter1 = plt.scatter(prompt_2d[:, 0], prompt_2d[:, 1], c=cluster_ids, 
					   cmap='tab20', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter1, label='Cluster ID')
plt.title('t-SNE Visualization of Prompts (Colored by Cluster ID)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('/home/ubuntu/ValueLearningInGenAI/vsl-rm/tsne_cluster_ids.png', dpi=300)
print("Saved: tsne_cluster_ids.png")

# Plot 2: Colored by source
plt.figure(figsize=(12, 8))
source_ids: list[int]	 = [source_colors[src] for src in sources]
scatter2= plt.scatter(prompt_2d[:, 0], prompt_2d[:, 1], c=source_ids,  # pyright: ignore[reportPrivateImportUsage]
					   cmap='tab20', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
cbar: Colorbar = plt.colorbar(scatter2, label='Source Index')
cbar.set_ticks(list(range(len(unique_sources))))
cbar.set_ticklabels([f'{i}: {src}' for i, src in enumerate(unique_sources)], fontsize=8)
plt.title('t-SNE Visualization of Prompts (Colored by Source)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('/home/ubuntu/ValueLearningInGenAI/vsl-rm/tsne_sources.png', dpi=300)
print("Saved: tsne_sources.png")

print("Done!")