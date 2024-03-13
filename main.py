#main.py
import pandas as pd 
from data_preprocessing.data_loader import load_data
from data_preprocessing.data_cleaner import preprocess_data
from models.model_manager import complete_analysis_pipeline 
from models.anomaly_detection import anomaly_detection_optimized

def process_file(file_path, save_dir=None, impute_missing_values=False):
    # Assuming `save_dir` might be used later for saving processed files
    raw_data = load_data(file_path)
    if raw_data is not None:
        processed_data, normalized_data = preprocess_data(raw_data, impute_missing_values)
        apply_analysis = complete_analysis_pipeline(processed_data)

# If you want to process a default file when main.py is run directly:
if __name__ == "__main__":
    default_file_path = 'banana_quality.csv'
    default_save_dir = '/Users/charlierobinson/Documents/Code/DissertationCode/Project 2/uploads'  # Example save directory, adjust as needed
    impute_missing_values_default = False  # Assuming you want the default to be False

    # Ensure the directory exists or create it
    import os
    if not os.path.exists(default_save_dir):
        os.makedirs(default_save_dir)

    process_file(default_file_path, default_save_dir, impute_missing_values_default)


