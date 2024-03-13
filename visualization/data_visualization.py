import os
import altair as alt
import pandas as pd

def altair_visualize_dimensionality_reduction_and_clustering(reduced_data, labels, method_name, feature_names):
    # Ensure labels are of type string for Altair visualization
    labels_str = labels.astype(str)

    # Create a DataFrame from your dimensionality-reduced data and labels
    # Assuming 'reduced_data' is a NumPy array; if it's already a DataFrame, adjust accordingly
    df = pd.DataFrame(reduced_data, columns=feature_names)
    df['Cluster'] = labels_str

    # Dynamically set the column names for the x and y axes
    x_col = feature_names[0] if len(feature_names) > 0 else 'Dim1'
    y_col = feature_names[1] if len(feature_names) > 1 else 'Dim2'

    # Create the Altair chart
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(f'{x_col}:Q', axis=alt.Axis(title=x_col)),
        y=alt.Y(f'{y_col}:Q', axis=alt.Axis(title=y_col)),
        color=alt.Color('Cluster:N', legend=alt.Legend(title="Cluster")),
        tooltip=[alt.Tooltip(f'{x_col}:Q'), alt.Tooltip(f'{y_col}:Q'), 'Cluster:N']
    ).properties(
        title=f'{method_name} Dimensionality Reduction with Clustering',
        width=600,
        height=400
    ).interactive()

    return chart


import altair as alt
import pandas as pd
import numpy as np

# Assuming reduced_data is the output from your dimensionality reduction process
# and is passed to anomaly_detection_optimized
# anomalies_data, normal_data, predictions = anomaly_detection_optimized(reduced_data)

def visualize_anomalies_with_predictions(reduced_data, predictions, filename='anomaly_chart.html'):
    # Ensure predictions is a flat, 1D array
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if predictions.ndim != 1:
        raise ValueError("Predictions array must be 1D")
    
    # Convert reduced_data to a DataFrame if it isn't already one
    if not isinstance(reduced_data, pd.DataFrame):
        reduced_data_df = pd.DataFrame(reduced_data, columns=['Dim1', 'Dim2'])
    else:
        reduced_data_df = reduced_data

    # Add predictions to the DataFrame
    reduced_data_df['Type'] = ['Anomaly' if pred == -1 else 'Normal' for pred in predictions.flatten()]
    
    # Create an Altair chart
    chart_anomaly = alt.Chart(reduced_data_df).mark_circle(size=60).encode(
        x='Dim1',
        y='Dim2',
        color=alt.Color('Type', scale=alt.Scale(domain=['Normal', 'Anomaly'], range=['blue', 'red'])),
        tooltip=['Dim1', 'Dim2', 'Type']
    ).properties(
        title='Anomaly Detection in Reduced Dimensionality Space',
        width=1000,  # Specify the width here
        height=800  # Specify the height here
    ).interactive()

    chart_anomaly.save(filename)
