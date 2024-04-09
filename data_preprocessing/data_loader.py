import pandas as pd

def load_data(file):
    # Debugging: Print the file type to help diagnose issues
    print(f"Loading file: {file.name}, type: {type(file)}")

    try:
        # Determine the file extension and choose the loading method accordingly
        if str(file.name).endswith('.csv'):
            # Attempt to read a CSV file
            try:
                return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(file, encoding='ISO-8859-1')  # Try latin1 encoding
                except UnicodeDecodeError:
                    return pd.read_csv(file, encoding='cp1252')  # Try Windows-1252 encoding
            except pd.errors.EmptyDataError:
                print("The file is empty. Please upload a valid file.")
                return None
        elif str(file.name).endswith('.xlsx'):
            return pd.read_excel(file)
        elif str(file.name).endswith('.json'):
            return pd.read_json(file)
        elif str(file.name).endswith('.txt'):
            return pd.read_csv(file, delimiter='\t')
    except Exception as e:
        # Catch any other exception and log it
        print(f"Failed to load file due to: {e}")
        return None
