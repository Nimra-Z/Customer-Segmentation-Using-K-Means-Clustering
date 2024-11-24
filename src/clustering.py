from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os

def perform_kmeans(data, n_clusters):
    """
    Performs KMeans clustering and calculates the silhouette score.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    clusters = kmeans.predict(data)
    score = silhouette_score(data, kmeans.labels_)
    return kmeans, clusters, score

def save_model(model, filepath):
    """
    Saves the KMeans model to a file.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the model
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

def load_model(filepath):
    """
    Loads a KMeans model from a file.
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)