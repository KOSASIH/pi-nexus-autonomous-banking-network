import pandas as pd
from sklearn.cluster import KMeans

class CustomerSegmentation:
    def __init__(self, customer_data):
        self.customer_data = customer_data

    def segment_customers(self):
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(self.customer_data)
        labels = kmeans.labels_
        return labels

# Example usage:
customer_segmentation = CustomerSegmentation(customer_data)
labels = customer_segmentation.segment_customers()
print(labels)
