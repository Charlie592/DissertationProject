import os
import altair as alt
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)



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

# Assuming reduced_data is the output from your dimensionality reduction process
# and is passed to anomaly_detection_optimized
# anomalies_data, normal_data, predictions = anomaly_detection_optimized(reduced_data)

def visualize_anomalies_with_predictions(reduced_data, predictions, filename='anomaly_chart_predictions.html'):
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


def plot_distributions_altair(data, columns, plot_type='boxplot', title=None, filename='anomaly_chart.html'):
    print("plotting columns {}".format(list(columns)))

    if plot_type not in {'boxplot', 'kdeplot'}:
        print("plot_type= {boxplot, kdeplot} only are supported")
        return
    
    # Ensure that data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    charts = []
    for col in columns:
        if plot_type == 'boxplot':
            chart = alt.Chart(data).mark_boxplot().encode(
                x=col + ':Q'
            ).properties(
                title=col
            )
        elif plot_type == 'kdeplot':
            chart = alt.Chart(data).transform_density(
                density=col,
                as_=[col, 'density'],
            ).mark_area().encode(
                x=col + ':Q',
                y='density:Q'
            ).properties(
                title=col
            )
        charts.append(chart)
    
    # Combine charts into a single visualization
    combined = alt.hconcat(*[alt.vconcat(*charts[i:i+4]) for i in range(0, len(charts), 4)])
    
    if title:
        combined = combined.properties(title=title)
    
    combined.save(filename)
    print(f"Plot saved as {filename}")



def plot_categorical_barcharts(data, categorical_columns, title=None, filename='categorical_plot_with_tooltips.html'):
    # Ensure that data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Filter only categorical columns for plotting
    categorical_data = data[categorical_columns].select_dtypes(include=['object', 'category'])

    charts = []
    for col in categorical_data.columns:
        # Create the bar chart with tooltips
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{col}:N", sort='-y'),  # N indicates a nominal (categorical) field
            y=alt.Y('count()', title='Count'),  # Count the number of entries for each category
            tooltip=[col, alt.Tooltip('count()', title='Count')]  # Show the tooltip when hovering
        ).properties(
            title=f"Bar Chart of {col}"
        )
        charts.append(chart)

    # Combine charts into a single visualization
    combined = alt.hconcat(*[alt.vconcat(*charts[i:i+3]) for i in range(0, len(charts), 3)])  # Adjust the number per row as needed

    if title:
        combined = combined.properties(title=title)
    
    combined.save(filename)
    print(f"Bar chart with tooltips saved as {filename}")

def plot_financial_barcharts(data, categorical_cols, financial_cols, title=None):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Start with an empty horizontal chart
    hconcat_charts = alt.HConcatChart(hconcat=[])
    
    for financial_col in financial_cols:
        # Initialize an empty list for the individual charts
        individual_charts = []
        for cat_col in categorical_cols:
            if cat_col == financial_col or cat_col not in data.columns:
                continue  # Skip if the same as the financial column or does not exist in the DataFrame

            # Properly aggregate the financial data
            aggregated_data = data.groupby(cat_col).agg({financial_col: 'sum'}).reset_index()

            # Create the bar chart
            chart = alt.Chart(aggregated_data).mark_bar().encode(
                x=alt.X(f'{cat_col}:N', title=cat_col),
                y=alt.Y(f'{financial_col}:Q', title=f'Sum of {financial_col}'),
                color=alt.Color(f'{cat_col}:N'),  # Here we remove the legend title
                tooltip=[alt.Tooltip(f'{cat_col}:N'), alt.Tooltip(f'{financial_col}:Q')]
            ).properties(
                title=f'Sum of {financial_col} by {cat_col}'
            )
            individual_charts.append(chart)

        # Combine charts for the current financial column side by side
        combined_charts_for_col = alt.hconcat(*individual_charts).resolve_scale(color='independent')
        
        # Add the combined chart for the current financial column to the overall horizontal concatenation
        hconcat_charts |= combined_charts_for_col

    if title:
        # Add title to the whole concatenated chart
        hconcat_charts = hconcat_charts.properties(title=title)

    return hconcat_charts
