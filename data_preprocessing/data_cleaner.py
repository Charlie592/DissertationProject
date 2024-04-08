# datacleaner.py

from tpot import TPOTRegressor, TPOTClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
from tpot import TPOTRegressor, TPOTClassifier


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import re 


def preprocess_data(data, handle_missing_values):
    normalized_data = data.copy()

    if handle_missing_values == False:
        print("Dropping rows with missing values.")
        num_rows_dropped = len(data) - len(data.dropna())
        data.dropna(inplace=True)
        normalized_data.dropna(inplace=True)
        print(f"Dropped {num_rows_dropped} rows with missing values.")

    financial_cols = detect_financial_columns(data)
    time_date_cols, converted_data = detect_time_date_columns(data)
    data = converted_data
    #print("Time Date Columns:", time_date_cols)
    #print(data.dtypes)
    categorical_cols = data.select_dtypes(include=['object']).columns
    #print("Categorical columns:", categorical_cols)

    for col in data.columns:
        if col in financial_cols:
            # Handle financial data specifically (e.g., normalization, categorization)
            #print(f"Handling financial column: {col}")
            continue
    
   
        # Initialize encoders 
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        label_encoder = LabelEncoder()

        for col in normalized_data.columns:
            if normalized_data[col].dtype == 'object':
                # Fill NaN values in categorical columns with a placeholder
                normalized_data[col].fillna('missing', inplace=True)

                # Determine encoding strategy based on the number of unique values
                unique_values = normalized_data[col].nunique()
                if unique_values >= 5:
                    # OneHot encode if unique values are 5 or less
                    encoded = one_hot_encoder.fit_transform(normalized_data[[col]])
                    encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out([col]), index=normalized_data.index)
                    normalized_data = pd.concat([normalized_data.drop(columns=[col]), encoded_df], axis=1)
                else:
                    # Label encode if unique values are more than 5
                    normalized_data[col] = label_encoder.fit_transform(normalized_data[col])
                
                

    else:
        normalized_data = handle_missing_values_with_tpot(normalized_data)
        normalized_data = normalize_data(normalized_data)
        return data, normalized_data, financial_cols, categorical_cols, time_date_cols

def detect_financial_columns(data):
    financial_keywords = ['revenue', 'cost', 'profit', 'expense', 'income', 'gross', 'salary', 'dollar', 'dollars', 'euro', 'pound', 'pounds', 'sterling', 'yen', 'rupee', 'ruble', 'real', 'peso', 'franc', 'lira', 'rand', 'krona', 'won', 'yuan', 'renminbi', 'rupee', 'ruble', 'real', 'peso', 'franc', 'lira', 'rand', 'krona', 'won', 'yuan', 'renminbi',]
    financial_cols = []
    for col in data.columns:
        # Consider both column name and the presence of currency symbols in the data
        if any(keyword in col.lower() for keyword in financial_keywords) or data[col].astype(str).str.contains(r'[\$\£\€]', regex=True).any():
            financial_cols.append(col)
    return financial_cols

def handle_missing_values_with_tpot(data):
    # Identify columns with missing values
    columns_with_missing_values = data.columns[data.isnull().any()].tolist()
    #print("\nColumns with missing values before imputation:", columns_with_missing_values)

    # Loop over columns with missing values and apply predictive imputation
    for column in columns_with_missing_values:
        data = predictive_imputation(data, column)
    return data

"""def detect_time_date_columns(data):
    time_date_keywords = ['date', 'time', 'hour', 'minute', 'second', 'day', 'month', 'year']
    time_date_cols = []
    for col in data.columns:
        # Consider both column name and the presence of time/date keywords
        if any(keyword in col.lower() for keyword in time_date_keywords):
            time_date_cols.append(col)
    return time_date_cols
"""

def detect_time_date_columns(data):
    time_date_keywords = ['date', 'time', 'hour', 'minute', 'second', 'day', 'month', 'year']
    time_date_cols = []
    
    for col in data.columns:
        # Check if column name contains any time/date keyword
        if any(keyword in col.lower() for keyword in time_date_keywords):
            try:
                # Try to convert to datetime
                data[col] = pd.to_datetime(data[col])
                time_date_cols.append(col)
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                # If conversion fails, it might be a time-only column or not a datetime column
                try:
                    # If it could be a time-only column, convert only the time component
                    if 'time' in col.lower() or 'hour' in col.lower():
                        data[col] = pd.to_datetime(data[col], format='%H:%M:%S').dt.time
                        time_date_cols.append(col)
                except (ValueError, TypeError):
                    # If conversion fails again, it's not a time column
                    continue
    return time_date_cols, data


def predictive_imputation(data, column_to_impute):
    # Identify and drop columns where all values (except for the header) are NaN
    non_na_counts = data.count()
    empty_columns = non_na_counts[non_na_counts == 0].index.tolist()
    data = data.drop(columns=empty_columns)
    
    # Check if the target column for imputation was dropped or is empty
    if column_to_impute not in data.columns:
        print(f"Column '{column_to_impute}' was empty and has been dropped.")
        return data
    
    missing_before = data[column_to_impute].isnull().sum()
    if missing_before == 0:
        print(f"No missing values in '{column_to_impute}'. No imputation needed.")
        return data
    
    if pd.api.types.is_numeric_dtype(data[column_to_impute]):
        tpot_model = TPOTRegressor(max_time_mins=30, generations=3, population_size=50, verbosity=2, random_state=42)
        print(f"Using TPOTRegressor for numeric column: {column_to_impute}")
    else:
        tpot_model = TPOTClassifier(max_time_mins=30, generations=5, population_size=50, verbosity=2, random_state=42)
        print(f"Using TPOTClassifier for categorical column: {column_to_impute}")

    # Splitting the data into training and testing sets based on null values in the column to impute
    train_data = data.dropna(subset=[column_to_impute])
    test_data = data[data[column_to_impute].isnull()]
    X_train = train_data.drop(columns=[column_to_impute])
    y_train = train_data[column_to_impute]
    X_test = test_data.drop(columns=[column_to_impute])

    # Fitting the model and predicting the missing values
    tpot_model.fit(X_train, y_train)
    predicted_values = tpot_model.predict(X_test)
    data.loc[data[column_to_impute].isnull(), column_to_impute] = predicted_values
    missing_after = data[column_to_impute].isnull().sum()

    print(f"Imputed missing values in '{column_to_impute}'. Missing before: {missing_before}, Missing after: {missing_after}")
    return data


def normalize_data(normalized_data):
    scaler = StandardScaler()
    numerical_cols = normalized_data.select_dtypes(include=['float64', 'int64']).columns
    normalized_data[numerical_cols] = scaler.fit_transform(normalized_data[numerical_cols])
    #print("Normalized data:\n", normalized_data.head())
    
    return normalized_data


def remove_outliers(data):
    """
    Remove outliers from numerical columns using IQR method.
    
    Parameters:
    - data (DataFrame): The DataFrame to clean.
    
    Returns:
    - DataFrame: DataFrame with outliers removed.
    """
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

