# predictor.py
from sklearn.linear_model import LinearRegression
import pandas as pd

"""def make_predictions(data, column_name):
    # Example: Simple Linear Regression for a numerical prediction
    # This is a placeholder; you'll want to replace this with your actual prediction logic
    
    # For demonstration, let's predict a column based on all others
    X = data.drop(column_name, axis=1)
    y = data[column_name]
    
    # Fit a regression model (you might have a more complex pipeline)
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    
    # Calculate some metrics (R^2 for regression)
    metrics = {
        'R^2': model.score(X, y)
    }
    
    return predictions, metrics
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def make_predictions(data, target_column):
    # Split data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = LinearRegression()
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create a DataFrame with actual and predicted values
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # Add a column for residuals
    predictions_df['Residual'] = predictions_df['Actual'] - predictions_df['Predicted']
    
    # Return the DataFrame and metrics
    return predictions_df, {'MSE': mse, 'R2': r2}
