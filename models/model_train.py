from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd



"""def train_the_model(processed_data, target_column):
    # Check if the target column exists in the dataframe
    if target_column not in processed_data.columns:
        raise ValueError(f"The target column '{target_column}' does not exist in the dataframe.")

    # Drop the target column dynamically based on the provided target_column name
    X = processed_data.drop(target_column, axis=1)
    y = processed_data[target_column]

    print("Total samples in X:", X.shape[0])
    print("Total samples in y:", y.shape[0])

    # Adjust the train_test_split to handle the dataset size dynamically
    test_size = 0.2 if len(X) > 1000 else 0.3  # Example condition, adjust based on your data size or requirements

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("Training samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])

    # Initialize TPOTClassifier or TPOTRegressor based on the target variable type
    if pd.api.types.is_numeric_dtype(y):
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
        print("Using TPOTRegressor for numeric target.")
    else:
        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
        print("Using TPOTClassifier for categorical target.")

    tpot.fit(X_train, y_train)
    print("TPOT accuracy: ", tpot.score(X_test, y_test))

    tpot.export('best_model_pipeline.py')"""

# Example usage:
# processed_data = pd.read_csv('your_dataset.csv')
# train_model(processed_data, 'your_target_column')





"""def train_model(processed_data):
    # Assuming `processed_data` is your preprocessed dataset and `target` is the target variable
    X = processed_data.drop('target', axis=1)
    y = processed_data['target']

    print("Total Asamples:", X.shape[0])
    print("Total Bsamples:", y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, test_size=3000, random_state=42)
    print("Total Csamples:", X_train.shape[0])
    print("Total Dsamples:", y_train.shape[0])
    print("Total Csamples:", X_test.shape[0])
    print("Total Dsamples:", y_test .shape[0])

    tpot = TPOTClassifier(generations=1, population_size=10, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    print("TPOT accuracy: ", tpot.score(X_test, y_test))
    tpot.export('best_model_pipeline.py')

"""