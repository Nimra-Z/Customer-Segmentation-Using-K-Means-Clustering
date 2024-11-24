import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(data):
    """
    Dynamically preprocesses the input dataset:
    - Handles missing values.
    - Standardizes numerical columns.
    - Encodes categorical columns.
    """
    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Separate numerical and categorical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns

    # Standardize numerical features
    scaler = StandardScaler()
    if len(numerical_features) > 0:
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Encode categorical features
    if len(categorical_features) > 0:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(data[categorical_features])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        encoded_df.index = data.index
        data = pd.concat([data.drop(categorical_features, axis=1), encoded_df], axis=1)

    # Debugging output
    print("Returning Values:")
    print("Processed Data (head):\n", data.head())
    print("Scaler object:", scaler)
    print("Numerical Features:", numerical_features)

    # Return processed data, scaler, and numerical feature names
    return data, scaler, numerical_features