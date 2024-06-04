import numpy as np
from sklearn.cluster import KMeans

# Load node data
node_data =...

# Create k-means model
kmeans = KMeans(n_clusters=8, random_state=42)

# Fit the model
kmeans.fit(node_data)

# Predict node clusters
node_clusters = kmeans.predict(node_data)
