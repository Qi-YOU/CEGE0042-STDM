import os
import gc
import pandas as pd


def detect_environment():
    """
    Automatically detect the current environment (colab, kaggle, or sys).
    
    Returns:
        str: The detected environment ('colab', 'kaggle', or 'sys').
    """

    if 'COLAB_GPU' in os.environ:
        return 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    else:
        return 'sys'  # Default to local system


def load_data(file_type):
    """
    Load the specified CSV file ('train', 'test' or 'data')
    based on the detected environment.
    
    Args:
        file_type (str): 'train', 'test' or 'data'.
    
    Returns:
        pandas.DataFrame: Loaded data as a DataFrame.
    """

    if file_type not in ['train', 'test', 'data']:
        raise ValueError("Invalid file_type. Use 'train', 'test' or 'data'.")

    # Detect the environment
    environment = detect_environment()
    
    # Define file paths for different environments
    file_paths = {
        'colab': {
            'data': '/content/data/data.csv',
            'train': '/content/data/train.csv',
            'test': '/content/data/test.csv'
        },
        'kaggle': {
            'data': '/kaggle/input/data/data.csv',
            'train': '/kaggle/input/data/train.csv',
            'test': '/kaggle/input/data/test.csv'
        },
        'sys': {
            'data': os.path.join(os.getcwd(), 'data', 'data.csv'),
            'train': os.path.join(os.getcwd(), 'data', 'train.csv'),
            'test': os.path.join(os.getcwd(), 'data', 'test.csv')
        }
    }

    # Find the file path according to environment-specific settings
    file_path = file_paths[environment][file_type]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type}.csv not found at {file_path}")

    # Load the CSV file
    print(f"Detected environment: {environment}")
    print(f"Loading {file_type}.csv from: {file_path}")
    return pd.read_csv(file_path)


def save_data(X, y, filename, folder="data"):
    """
    Saves the given features (X) and target (y) as a CSV file in the specified folder.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series or pd.DataFrame): Target labels.
    - filename (str): Name of the output CSV file (e.g., 'train.csv' or 'test.csv').
    - folder (str): Folder where the file will be saved. Default is 'data'.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Concatenate features and target
    data = pd.concat([X, y], axis=1)
    
    # Save to CSV
    file_path = os.path.join(folder, filename)
    data.to_csv(file_path, index=False)
    
    print(f"Saved {filename} successfully to: {file_path}")


if __name__ == "__main__":
    # Example usage when running this script directly
    try:
        # Load the test data
        data = load_data("data")
        print(data.head())
 
        # Load the train data
        train_data = load_data("train")
        print(train_data.head())

        del train_data
        gc.collect()

        # Load the test data
        test_data = load_data("test")
        print(test_data.head())
    except Exception as e:
        print(f"Error: {e}")
