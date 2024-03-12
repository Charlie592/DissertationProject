import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    """
    Load data from a specified file path using Pandas, then extract dataset characteristics.
    
    Parameters:
    - file_path (str): The path to the dataset file.
    
    Returns:
    - DataFrame: The loaded data as a Pandas DataFrame.
    - dict: Extracted characteristics of the dataset.
    """
    try:
        data = None
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            print("Unsupported file format.")
            return None, None
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


