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
    Load the specified CSV file ('train' or 'test')
    based on the detected environment.
    
    Args:
        file_type (str): 'train' or 'test'.
    
    Returns:
        pandas.DataFrame: Loaded data as a DataFrame.
    """

    if file_type not in ['train', 'test']:
        raise ValueError("Invalid file_type. Use 'train' or 'test'.")

    # Detect the environment
    environment = detect_environment()
    
    # Define file paths for different environments
    file_paths = {
        'colab': {
            'train': '/content/data/train.csv',
            'test': '/content/data/test.csv'
        },
        'kaggle': {
            'train': '/kaggle/input/data/train.csv',
            'test': '/kaggle/input/data/test.csv'
        },
        'sys': {
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


if __name__ == "__main__":
    # Example usage when running this script directly
    try:
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
