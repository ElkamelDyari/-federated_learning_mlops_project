import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

warnings.filterwarnings("ignore")


# --- Functions for Data Processing ---

def load_data(file_path):
    """Load the dataset and remove duplicates."""
    data = pd.read_csv(file_path)
    data.drop_duplicates(inplace=True)
    return data


def clean_missing_values(data):
    """Replace infinities and fill NaNs with medians for specific columns."""
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    med_flow_bytes = data['Flow Bytes/s'].median()
    med_flow_packets = data['Flow Packets/s'].median()
    data['Flow Bytes/s'].fillna(med_flow_bytes, inplace=True)
    data['Flow Packets/s'].fillna(med_flow_packets, inplace=True)
    return data


def map_attack_types(data):
    """Map attack labels to their corresponding types and add binary target labels."""
    attack_map = {
        'BENIGN': 'BENIGN',
        'DDoS': 'DDoS',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'PortScan': 'Port Scan',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'Bot': 'Bot',
        'Web Attack-Brute Force': 'Web Attack',
        'Web Attack-XSS': 'Web Attack',
        'Web Attack-Sql Injection': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'Heartbleed',
    }

    binary_attack = {key: (0 if key == 'BENIGN' else 1) for key in attack_map.values()}

    data['Attack Type'] = data['Label'].map(attack_map)
    data['Target'] = data['Attack Type'].map(binary_attack)
    return data


def encode_attack_types(data):
    """Encode attack types to numerical values."""
    le = LabelEncoder()
    data['Attack Number'] = le.fit_transform(data['Attack Type'])
    return data


def optimize_memory_usage(data):
    """Downcast numerical columns to reduce memory usage."""
    for col in data.columns:
        col_type = data[col].dtype
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(
                    np.float32).max:
                data[col] = data[col].astype(np.float32)
            elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                data[col] = data[col].astype(np.int32)
    return data


def drop_unnecessary_features(data):
    """Drop features with only one unique value and unnecessary columns."""
    num_unique = data.nunique()
    valid_columns = num_unique[num_unique > 1].index
    data = data[valid_columns]
    data.drop(["Label", "Attack Type", "Attack Number"], axis=1, inplace=True, errors='ignore')
    return data



def split_data(data, n_splits=25, random_state=0):
    """Splits data into equal-sized batches.

      Args:
        data: The data to be split.
        n_splits: The number of splits.
        random_state: The random state for shuffling the data.

      Returns:
        A list of batches, each containing an equal number of samples from the data.
      """
    # Shuffle the data to ensure randomness
    np.random.seed(random_state)
    np.random.shuffle(data)

    # Calculate the batch size
    batch_size = len(data) // n_splits

    # Split the data into batches
    batches = []
    for i in range(n_splits):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch = data[start_idx:end_idx]
        batches.append(batch)

    return batches


def save_batches(batches, output_dir):
    """Save batches to CSV files."""
    for i, batch in enumerate(batches):
        file_path = f"{output_dir}/data{i + 1}.csv"
        batch.to_csv(file_path, index=False)




# Load and preprocess the data
file_path = "../../data/raw/MachineLearningCVE/MachineLearningCVE.csv"
data = load_data(file_path)
data = clean_missing_values(data)
data = map_attack_types(data)
data = encode_attack_types(data)
data = optimize_memory_usage(data)
data = drop_unnecessary_features(data)

# split the data
batches = split_data(data)

# Save processed batches
output_dir = "../../data/processed"
save_batches(batches, output_dir)
print("Data processing complete!")