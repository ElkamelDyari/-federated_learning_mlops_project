import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings("ignore")

# Preprocessing function
def preprocess_data(file_path, n_components=35, batch_size=500, test_size=0.3, random_state=0):
    dataset = pd.read_csv(file_path)
    features = dataset.drop('Target', axis=1)
    attacks = dataset['Target']

    # Create a pipeline for preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ipca', IncrementalPCA(n_components=n_components, batch_size=batch_size))
    ])

    # Fit the pipeline on the features
    pipeline.fit(features)

    # Transform the features
    transformed_features = pipeline.transform(features)
    new_data = pd.DataFrame(transformed_features, columns=[f'PC{i + 1}' for i in range(n_components)])
    new_data['Target'] = attacks.values

    # Save the pipeline to artifacts
    joblib.dump(pipeline, 'artifacts/preprocessing_pipeline.pkl')

    # Split data
    X_new = new_data.drop('Target', axis=1)
    y_new = new_data['Target']

    return train_test_split(X_new, y_new, test_size=test_size, random_state=random_state)

def process_pred(X, n_components=35, batch_size=500):
    # Load the saved pipeline
    pipeline = joblib.load('artifacts/preprocessing_pipeline.pkl')

    # Transform the features using the loaded pipeline
    transformed_features = pipeline.transform(X)
    new_data = pd.DataFrame(transformed_features, columns=[f'PC{i + 1}' for i in range(n_components)])

    return new_data