# Customer Segmentation

Customer Segmentation is a machine learning project designed to group customers into distinct segments based on their attributes such as demographics, spending behavior, and preferences. This project uses clustering algorithms like **KMeans** to identify customer groups and provides an interactive **Streamlit** web application for visualization and exploration.

---

## **Features**
1. **Dynamic Dataset Handling**:
   - Upload your own dataset in CSV format.
   - Automatically preprocesses the dataset, handling missing values, scaling, and encoding.

2. **Clustering**:
   - Uses **KMeans Clustering** to group customers into segments.
   - Calculates the **Silhouette Score** to evaluate clustering performance.

3. **Visualization**:
   - Reduces dimensionality using **PCA** for 2D visualization.
   - Interactive visualizations of customer clusters.

4. **Download Results**:
   - Allows downloading the clustering results with original feature values as a CSV file.


## **Technologies Used**
- **Python**: Core programming language.
- **Streamlit**: Web application framework for interactivity.
- **Scikit-learn**: For clustering, preprocessing, and PCA.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib**: Visualization library.

---

## **Directory Structure**

```
Customer_Segmentation/
├── app/
│   └── streamlit_app.py       # Streamlit application
├── src/
│   ├── preprocessing.py       # Preprocessing logic
│   ├── clustering.py          # Clustering logic
│   └── train_model.py         # Model training script
├── model/
│   └── kmeans_model.pkl       # Saved KMeans model
├── data/
│   └── Mall_Customer.csv      # Dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/Nimra-Z/Customer-Segmentation-Using-K-Means-Clustering.git
cd Customer-Segmentation-Using-K-Means-Clustering
```

### **2. Install Dependencies**
Create a virtual environment (optional) and install the required packages:
```bash
# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

### **4. Upload Dataset**
- Use the provided `Mall_Customer.csv` in the `data/` folder or upload your own dataset in CSV format.
- The dataset should contain numerical and/or categorical features.


## **Project Workflow**

1. **Upload Dataset**:
   - Load a dataset using the Streamlit file uploader.

2. **Data Preprocessing**:
   - Missing values are handled (numerical columns: median, categorical columns: mode).
   - Numerical features are standardized.
   - Categorical features are one-hot encoded.

3. **Clustering**:
   - KMeans clustering groups customers based on feature similarity.
   - The user can select the number of clusters dynamically.

4. **Visualization**:
   - PCA reduces dimensionality for 2D visualization of clusters.

5. **Download Results**:
   - The clustering results, including original-scale numerical values and cluster labels, can be downloaded as a CSV file.


## **Example Dataset**

An example dataset, `Mall_Customer.csv`, is provided in the `data/` directory. Here’s a preview:

| Gender  | Age | Annual Income (k$) | Spending Score (1-100) |
|---------|-----|---------------------|------------------------|
| Male    | 19  | 15                  | 39                     |
| Female  | 21  | 15                  | 81                     |
| Female  | 20  | 16                  | 6                      |
| Male    | 23  | 16                  | 77                     |
| Female  | 31  | 17                  | 40                     |


## **License**
This project is licensed under the MIT License. See the LICENSE file for details.
