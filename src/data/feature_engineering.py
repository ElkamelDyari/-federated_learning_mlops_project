import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


# Preprocessing function
def preprocess_data(file_path, n_components=35, batch_size=500, test_size=0.3, random_state=0):
    dataset = pd.read_csv(file_path)
    features = dataset.drop('Target', axis=1)
    attacks = dataset['Target']

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Incremental PCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for batch in np.array_split(scaled_features, len(features) // batch_size):
        ipca.partial_fit(batch)

    transformed_features = ipca.transform(scaled_features)
    new_data = pd.DataFrame(transformed_features, columns=[f'PC{i + 1}' for i in range(n_components)])
    new_data['Target'] = attacks.values

    # Split data
    X_new = new_data.drop('Target', axis=1)
    y_new = new_data['Target']

    return train_test_split(X_new, y_new, test_size=test_size, random_state=random_state)
