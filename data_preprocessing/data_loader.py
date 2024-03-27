import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pandas as pd

# Assuming load_data uses pandas to load the file
def load_data(file):
    # Check if file is a file-like object
    if hasattr(file, 'read'):
        # Use pandas to load file directly from file-like object
        if str(file.name).endswith('.csv'):
            return pd.read_csv(file)
        elif str(file.name).endswith('.xlsx'):
            return pd.read_excel(file)
        elif str(file.name).endswith('.json'):
            return pd.read_json(file)
        elif str(file.name).endswith('.txt'):
            return pd.read_csv(file, delimiter='\t')
    else:
        # Original load_data functionality for file paths
        # Your existing code to load data from a path
        pass


