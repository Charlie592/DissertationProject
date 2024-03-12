#anomaly_detection
from sklearn.ensemble import IsolationForest
import numpy as np

def anomaly_detection_optimized(data, score_percentile=95):
    # Initialize the Isolation Forest model
    iso_forest_model = IsolationForest(n_estimators=100, random_state=42)
    
    # Fit the model to your data
    iso_forest_model.fit(data)
    
    # Use decision_function to get anomaly scores (the lower, the more abnormal)
    scores = iso_forest_model.decision_function(data)
    
    # Determine a threshold for defining outliers based on the desired percentile of scores
    threshold = np.percentile(scores, 100 - score_percentile)
    
    # Predictions: -1 for anomalies, 1 for normal
    predictions = np.where(scores <= threshold, -1, 1)
    
    # Filter the dataset for anomalies or normal data based on the prediction
    anomalies_data = data[predictions == -1]
    normal_data = data[predictions == 1]
    
    # Print the number of anomalies detected
    print(f"Number of anomalies detected: {len(anomalies_data)}")
    
    return anomalies_data, normal_data, predictions
