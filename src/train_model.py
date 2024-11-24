import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.preprocessing import preprocess_data
from src.clustering import perform_kmeans, save_model

# Load dataset
data = pd.read_csv('data/Mall_Customers.csv')

# Preprocess data
processed_data = preprocess_data(data)

# Train KMeans model
n_clusters = 5  # Default number of clusters
kmeans, _, score = perform_kmeans(processed_data, n_clusters)

# Save the model
save_model(kmeans, 'model/kmeans_model.pkl')
print(f"Model trained and saved with Silhouette Score: {score:.2f}")