#main.py
import os
import pandas as pd 
from data_preprocessing.data_loader import load_data
from data_preprocessing.data_cleaner import preprocess_data
from models.model_manager import complete_analysis_pipeline 
from models.anomaly_detection import anomaly_detection_optimized
from visualization.data_visualization import plot_distributions_altair, plot_categorical_barcharts, plot_financial_barcharts

def process_file(file, save_dir=None, impute_missing_values=False):
    # Adjusted to handle file-like object directly
    raw_data = load_data(file)
    processed_data, normalized_data, financial_cols, categorical_cols = preprocess_data(raw_data, impute_missing_values)
    
    # Perform analysis (assuming this function now returns something, e.g., a summary)
    labels = complete_analysis_pipeline(processed_data, normalized_data)
    
    # Return processed data and any results or plots for Streamlit to display
    return processed_data, normalized_data, labels, financial_cols, categorical_cols

def process_visualizations():
    None


# If you want to process a default file when main.py is run directly:
if __name__ == "__main__":
    default_file_path = '/Users/charlierobinson/Documents/Code/DissertationCode/DissertationProject/uploads/train.csv'
    default_save_dir = '/Users/charlierobinson/Documents/Code/DissertationCode/DissertationProject/uploads'  # Example save directory, adjust as needed
    impute_missing_values_default = False  # Assuming you want the default to be False

    # Ensure the directory exists or create it
    if not os.path.exists(default_save_dir):
        os.makedirs(default_save_dir)

    process_file(default_file_path, default_save_dir, impute_missing_values_default)
