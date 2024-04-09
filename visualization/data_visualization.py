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

"""def visualize_anomalies_with_predictions(reduced_data, predictions, filename='anomaly_chart_predictions.html'):
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

    chart_anomaly.save(filename)"""


def plot_distributions_altair(data, plot_type='boxplot', title=None):
    # Select only numeric columns for plotting
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if plot_type not in {'boxplot', 'kdeplot'}:
        print("plot_type= {boxplot, kdeplot} only are supported")
        return

    charts = []
    for col in numeric_columns:
        if plot_type == 'boxplot':
            chart = alt.Chart(data).mark_boxplot().encode(
                x=alt.X(col + ':Q', title=col)  # Including title for each boxplot
            )
        elif plot_type == 'kdeplot':
            chart = alt.Chart(data).transform_density(
                density=col,
                as_=[col, 'density']
            ).mark_area().encode(
                x=alt.X(col + ':Q', title=col),  # Including title for the density plot
                y='density:Q'
            )
        charts.append(chart)
    
    # Combine charts into a single visualization, with a set number per row
    combined = alt.hconcat(*[alt.vconcat(*charts[i:i+4]) for i in range(0, len(charts), 3)])

    if title:
        combined = combined.properties(title=title)
    
    # Configure the chart with a dark theme
    combined = combined.configure(
        axis=alt.AxisConfig(
            labelColor='white',
            titleColor='white',
            gridColor='white',
            domainColor='white',
            tickColor='white',
            
        ),
        title=alt.TitleConfig(color='white')
    )

    return combined  # Return the combined chart object



def plot_categorical_barcharts(data, categorical_cols, title=None):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    individual_charts = []

    for col in categorical_cols:
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{col}:N", sort='-y'),
            y=alt.Y('count()', title='Count'),
            tooltip=[col, alt.Tooltip('count()', title='Count')]
        ).properties(
            title=f"Bar Chart of {col}"
            # Instead of setting width to 'container', we will let Streamlit handle it with use_container_width=True
        )
        individual_charts.append(chart)

    # Combine individual charts horizontally
    combined = alt.hconcat(*individual_charts).resolve_scale(
        x='independent', 
        y='independent'
    )

    if title:
        combined = combined.properties(title=title)
    
    return combined


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


def plot_time_series_charts(data, time_date_cols, numerical_cols, title=None):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Initialize an empty vertical chart
    vconcat_charts = alt.VConcatChart(vconcat=[])
    
    for time_col in time_date_cols:
        if time_col not in data.columns:
            continue  # Skip if the column does not exist in the DataFrame
        
        # Convert the time column to datetime if not already
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.sort_values(by=time_col)

        # Loop through each numerical column to create a separate line chart
        for num_col in numerical_cols:
            # Create the line chart
            chart = alt.Chart(data).mark_line(point=True).encode(
                x=alt.X(time_col, title='Date', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(num_col, title=num_col),
                tooltip=[alt.Tooltip(time_col), alt.Tooltip(num_col)]
            ).properties(
                title=f'{num_col} over time'
            ).interactive()
            
            vconcat_charts &= chart  # Concatenate charts vertically

    if title:
        # Add title to the whole concatenated chart
        vconcat_charts = vconcat_charts.properties(title=title)

    return vconcat_charts



# Function to create scatter plot without regression line
def scatter_plot(data, x_col, y_col):
    chart = alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X(x_col, type='quantitative', title=x_col),
        y=alt.Y(y_col, type='quantitative', title=y_col),
        tooltip=[x_col, y_col]
    )
    return chart

# Function to create scatter plot with regression line
def scatter_plot_with_regression(data, x_col, y_col):
    scatter_plot = alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X(x_col, type='quantitative', title=x_col),
        y=alt.Y(y_col, type='quantitative', title=y_col),
        tooltip=[x_col, y_col]
    )
    regression_line = scatter_plot.transform_regression(
        x_col, y_col, method="linear"
    ).mark_line(color='red').encode(
        x=alt.X(x_col, type='quantitative'),
        y=alt.Y('y:Q', title=y_col)
    )
    return scatter_plot + regression_line


def create_scatter_plot(data, x_col, y_col):
    return alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X(x_col, title=x_col),
        y=alt.Y(y_col, title=y_col),
        tooltip=[x_col, y_col]
    )

def create_scatter_plot_with_line(data, x_col, y_col):
    # Base chart for scatter points
    scatter_plot = alt.Chart(data).mark_circle(size=60).encode(
        alt.X(x_col, type='quantitative', title=x_col),
        alt.Y(y_col, type='quantitative', title=y_col),
        tooltip=[x_col, y_col]  # Tooltips on hover
    )

    # Regression line
    regression_line = scatter_plot.transform_regression(
        x_col, y_col, method="linear"
    ).mark_line(color='red')

    # Combine the scatter plot and the regression line
    final_chart = scatter_plot + regression_line

    return final_chart


def visualize_feature_relationships(data, labels, AI_response, features=None, save_figures=False, figures_dir='figures'):
    figures = []  # A list to store the matplotlib figure objects or figure paths if saved
    data_with_clusters = data.copy()  # Create a copy of the data
    data_with_clusters['Cluster'] = labels  # Add the Cluster column to the copied data

    # Example: Calculate the mean for numerical features for each cluster
    cluster_characteristics = data_with_clusters.groupby('Cluster').mean()

    # Identifying top distinguishing features for one cluster as an example
    top_features = cluster_characteristics.loc[0].sort_values(ascending=False)[:3].index.tolist()
    #print("Top distinguishing features for Cluster 0:", top_features)


    # Ensure the figures directory exists if saving figures
    if save_figures:
        import os
        os.makedirs(figures_dir, exist_ok=True)

    AI_response_fig={}
    for cluster in sorted(data_with_clusters['Cluster'].unique()):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        if features:
            cluster_data = cluster_data[features + ['Cluster']]  # Select specified features and Cluster column

        # Correlation heatmap
        heatmap_fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cluster_data.drop('Cluster', axis=1).corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
        plt.title(f"Feature Correlations in Cluster {cluster}")

        AI_response_fig[heatmap_fig]=AI_response[cluster]
        if save_figures:
            heatmap_path = f"{figures_dir}/heatmap_cluster_{cluster}.png"
            heatmap_fig.savefig(heatmap_path)
            figures.append(heatmap_path)
            plt.close(heatmap_fig)
        else:
            figures.append(heatmap_fig)
      
        

    return figures, AI_response_fig