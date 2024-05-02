# datacleaner.py

from tpot import TPOTRegressor, TPOTClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
from tpot import TPOTRegressor, TPOTClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from dateutil import parser
import re


def preprocess_data(data, handle_missing_values):

    data = drop_id_columns(data)
    financial_cols = detect_financial_columns(data)
    time_date_cols, converted_data, converted_to_normalize = detect_time_date_columns(data)
    data = converted_data
    print("Time/Date columns:", time_date_cols)
    print(data.dtypes)
    print(data.head(10))
    print(converted_to_normalize.dtypes)
    categorical_cols = data.select_dtypes(include=['object']).columns
    #print("Categorical columns:", categorical_cols)

    normalized_data = converted_to_normalize.copy()
    print("Normalized data:\n", normalized_data.head(10))

    if handle_missing_values == False:
        print("Dropping rows with missing values.")
        num_rows_dropped = len(data) - len(data.dropna())
        data.dropna(inplace=True)
        normalized_data.dropna(inplace=True)
        print(f"Dropped {num_rows_dropped} rows with missing values.")

   
        # Initialize encoders 
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        label_encoder = LabelEncoder()

        for col in normalized_data.columns:
            if normalized_data[col].dtype == 'object':
                # Fill NaN values in categorical columns with a placeholder
                normalized_data[col].fillna('missing', inplace=True)

                # Determine encoding strategy based on the number of unique values
                unique_values = normalized_data[col].nunique()
                if unique_values <= 5:
                    # OneHot encode if unique values are 5 or less
                    encoded = one_hot_encoder.fit_transform(normalized_data[[col]])
                    encoded_data = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out([col]), index=normalized_data.index)
                    normalized_data = pd.concat([normalized_data.drop(columns=[col]), encoded_data], axis=1)
                else:
                    # Label encode if unique values are more than 5
                    normalized_data[col] = label_encoder.fit_transform(normalized_data[col])
                
                

    else:
        normalized_data = handle_missing_values_with_tpot(normalized_data)
        normalized_data = normalize_data(normalized_data)
    
    return data, normalized_data, financial_cols, categorical_cols, time_date_cols

def detect_financial_columns(data):
    financial_keywords = [
        'revenue', 'cost', 'profit', 'expense', 'income', 'gross', 'salary',
        'dollar', 'dollars', 'euro', 'pound', 'pounds', 'sterling', 'yen',
        'rupee', 'ruble', 'real', 'peso', 'franc', 'lira', 'rand', 'krona',
        'won', 'yuan', 'renminbi', 'euros', 'pounds', 'dollars', 'rupees',
    ]

    financial_cols = []
    for col in data.columns:
        # Check for full-word match in column name
        if any(re.search(r'\b{}\b'.format(keyword), col.lower()) for keyword in financial_keywords):
            financial_cols.append(col)
            continue  # If the keyword is found in the column name, no need to check for symbols in its data
        
        # Check for presence of currency symbols in the data of the column
        if data[col].astype(str).str.contains(r'[\$\£\€]', regex=True).any():
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

def detect_time_date_columns(data):
    time_date_cols = []
    data2norm = data.copy()
    date_format_YYYY = {
        "%Y-%m-%d": r'\d{4}\D\d{1,2}\D\d{1,2}'  # YYYY-M-D or YYYY-MM-DD
    }
    date_formats = {
        "%d/%m/%Y": r'\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}',  # D/M/YYYY or DD/MM/YYYY
        "%m/%d/%Y": r'\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}'   # M/D/YYYY or MM/DD/YYYY
    }
    time_pattern = r'\d{2}:\d{2}(:\d{2})?(\sAM|\sPM)?'


    def infer_date_format(date_str):
        for format, regex in date_formats.items():
            if re.match(regex, date_str):
                return format
        return None  # Unknown format
    
    def parse_time(time_str):
        for fmt in ('%H:%M', '%I:%M:%S %p', '%H:%M:%S'):
            try:
                # Parse the time string using the given format
                parsed_time = pd.to_datetime(time_str, format=fmt)
                # Return only the hour part as a string
                return str(parsed_time.hour).zfill(2)  # zfill(2) ensures a two-digit format, like '02', '11'
            except ValueError:
                # If parsing fails, continue to the next format
                continue
        return None 

    def parse_time2norm(time_str):
        for fmt in ('%H:%M', '%I:%M:%S %p', '%H:%M:%S'):
            try:
                # Return only the time part
                return pd.to_datetime(time_str, format=fmt).time()
            except ValueError:
                continue
        return None # or raise an exception
    

    for col in data.columns:
        sample_values = data[col].dropna().astype(str).sample(min(100, len(data[col])))
        # Flag to track if column has been processed as date or time
        processed_as_date_or_time = False

        # Check if any sample value matches the YYYY date format
        if any(re.search(pattern, value) for pattern in date_format_YYYY.values() for value in sample_values):
            # Process the column as a date
            data[f'{col}_date'] = pd.to_datetime(data[col], format='%Y-%m-%d', errors='coerce')
            data2norm[f'{col}_date'] = pd.to_datetime(data[col], format='%Y-%m-%d', errors='coerce').dt.strftime('%Y-%m-%d')
            # Mark the column as processed
            processed_as_date_or_time = True
            time_date_cols.append(f'{col}_date')
            
        # Check if any sample value matches any of the other date formats
        elif any(re.search(pattern, value) for pattern in date_formats.values() for value in sample_values):
            # Further processing for other date formats
            sample_dates = [value for value in sample_values if any(re.search(date_formats[fmt], value) for fmt in date_formats)]
            if sample_dates:
                # Infer the date format based on the first sample date
                inferred_format = infer_date_format(sample_dates[0])
                if inferred_format:
                    # Process the column according to the inferred format
                    data[f'{col}_date'] = pd.to_datetime(data[col], format=inferred_format, errors='coerce')
                    data2norm[f'{col}_date'] = pd.to_datetime(data[col], format=inferred_format, errors='coerce').dt.strftime('%Y-%m-%d')
                    # Mark the column as processed
                    processed_as_date_or_time = True
                    time_date_cols.append(f'{col}_date')

        
        if any(re.search(time_pattern, value) for value in sample_values):
            # Process as time
            data[f'{col}_time'] = data[col].apply(parse_time)
            data2norm[f'{col}_time'] = data[col].apply(parse_time2norm)
            # Mark column as processed
            processed_as_date_or_time = True
            time_date_cols.append(f'{col}_time')
        
        # If the column was processed as date or time, drop the original column
        if processed_as_date_or_time:
            data.drop(col, axis=1, inplace=True)
            data2norm.drop(col, axis=1, inplace=True)


    # Assuming the function should return the modified DataFrame and the list of new time/date related columns
    return time_date_cols, data, data2norm



import pandas as pd

def drop_id_columns(data):
    # Define keywords for identifying ID columns
    id_keywords = ['_id', 'id_', ' id', 'id ']

    # Initialize list to store ID columns
    id_columns = []

    # Iterate over the columns of the DataFrame
    for col in data.columns:
        # Check for full-word match in column name
        if any(re.search(r'\b{}\b'.format(keyword), col.lower()) for keyword in id_keywords):
            # If full-word match is found, add the column to the list of ID columns
            id_columns.append(col)
        
        # Check if the data in the column increments by 1 row by row
        elif pd.api.types.is_numeric_dtype(data[col]):
            is_id_column = True
            prev_value = None
            for value in data[col]:
                if prev_value is not None and value != prev_value + 1:
                    is_id_column = False
                    break
                prev_value = value
            
            if is_id_column:
                id_columns.append(col)

        #if len(data[col]) == data[col].nunique():
            #id_columns.append(col)

    # Print information about the columns and ID columns
    print(f"Columns: {data.columns}")
    print(f"Dropping ID columns: {id_columns}")

    # Drop the ID columns from the DataFrame, ignoring errors
    data_no_id = data.drop(columns=id_columns, errors='ignore')

    # Return the modified DataFrame
    return data_no_id



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
