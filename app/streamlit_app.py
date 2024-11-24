import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data
from src.clustering import perform_kmeans, save_model

# Streamlit Page Configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# App Title
st.title("Customer Segmentation")
st.write("Upload your dataset, and the app will preprocess it and perform clustering using KMeans.")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Original Dataset")
    st.dataframe(data.head())

    # Preprocess data
    processed_data, scaler, numerical_features = preprocess_data(data)
    st.write("### Preprocessed Data")
    st.dataframe(processed_data.head())

    # Select the number of clusters
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

    # Perform clustering
    if st.button("Run K-Means Clustering"):
        kmeans, clusters, score = perform_kmeans(processed_data, n_clusters)

        # Save the model
        save_model(kmeans, 'model/kmeans_model.pkl')
        st.write("Model trained and saved!")

        # Add cluster labels to the processed data
        processed_data['Cluster'] = clusters

        # Inverse transform numerical features to original scale
        processed_data[numerical_features] = scaler.inverse_transform(processed_data[numerical_features])

        # Display clustering results
        st.write("### Clustering Results")
        st.dataframe(processed_data.head())

        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(processed_data.drop('Cluster', axis=1))
        pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = clusters

        st.write("### Cluster Visualization (PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster in pca_df['Cluster'].unique():
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {cluster}")
        ax.set_title("Customer Segments")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend()
        st.pyplot(fig)

        # Download clustering results
        csv_data = processed_data.to_csv(index=False)
        st.download_button(
            label="Download Clustering Results as CSV",
            data=csv_data,
            file_name="clustering_results_original_scale.csv",
            mime="text/csv"
        )