import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

def make_predictions(data, target_column, selected_features, selected_model, task_type, model_params):
    # Filter the dataset
    data = data[selected_features + [target_column]]
    
    # Split data into features and target
    X = data[selected_features]
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model initialization
    if selected_model == 'Linear Regression':
        model = LinearRegression()
    elif selected_model == 'Random Forest' and task_type == 'Regression':
        model = RandomForestRegressor(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=42)
    elif selected_model == 'Random Forest' and task_type == 'Classification':
        model = RandomForestClassifier(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=42)
    elif selected_model == 'Support Vector Machine' and task_type == 'Regression':
        model = SVR(C=model_params['C'], gamma=model_params['gamma'])
    elif selected_model == 'Support Vector Machine' and task_type == 'Classification':
        model = SVC(C=model_params['C'], gamma=model_params['gamma'], probability=True)
    
    # Fit the model
    model.fit(X_train, y_train)

    # Adjusting explainer initialization
    if selected_model.startswith("Support Vector"):
        background_data = shap.sample(X_train, 100)
        if isinstance(background_data, pd.DataFrame):
            background_data = background_data.values
        explainer = shap.KernelExplainer(model.predict, background_data)
        shap_values = explainer.shap_values(shap.sample(X_test, 10))
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

    # Correctly processing SHAP values
    shap_values_processed = np.abs(shap_values).mean(0) if isinstance(shap_values, np.ndarray) else np.abs(shap_values.values).mean(0)
    shap_summary_df = pd.DataFrame(shap_values_processed, index=selected_features, columns=['SHAP Value'])
    shap_summary_df.sort_values(by='SHAP Value', ascending=False, inplace=True)

    # Make predictions
    if task_type == 'Regression':
        y_pred = model.predict(X_test)
    elif task_type == 'Classification':
        # For SVC, predict_proba may not be available unless probability=True
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for the positive class
        else:
            y_pred_proba = None
        y_pred = model.predict(X_test)
    
    # Calculate metrics for regression
    if task_type == 'Regression':
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {'MSE': mse, 'MAE': mae, 'R² Score': r2}
    # Calculate metrics for classification
    elif task_type == 'Classification':
        accuracy = accuracy_score(y_test, y_pred)
        # Adjusting for multiclass classification
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    
    # Prepare the DataFrame
    if task_type == 'Regression':
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    elif task_type == 'Classification' and y_pred_proba is not None:
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Probability': y_pred_proba})
    else:
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # Return the predictions, metrics, and SHAP summary DataFrame
    return predictions_df, metrics, shap_summary_df
