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


def plot_categorical_barcharts(data, categorical_cols, N=20, min_count=3):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    individual_charts = []

    for col in categorical_cols:
        counts = data[col].value_counts().reset_index()
        counts.columns = [col, 'count']

        # Filter out categories with count less than min_count
        counts = counts[counts['count'] >= min_count]

        # Check the actual number of categories
        actual_categories_count = min(len(counts), N)
        title_text = f"Top {actual_categories_count} {col}" if actual_categories_count < N else f"Top {N} {col}"

        # If there are more than N categories, include "Other"
        if len(counts) > N:
            top_counts = counts.head(N-1)
            other_count = counts['count'][N-1:].sum()
            top_counts = top_counts.append({col: 'Other', 'count': other_count}, ignore_index=True)
            # Create a sort_key that places 'Other' at the end
            top_counts['sort_key'] = range(len(top_counts) - 1, -1, -1)
            top_counts.loc[top_counts[col] == 'Other', 'sort_key'] = -1
            top_counts = top_counts.sort_values('sort_key', ascending=False).drop(columns='sort_key')
        else:
            top_counts = counts

        if not top_counts.empty:
            chart = alt.Chart(top_counts).mark_bar(cornerRadius=3).encode(
                x=alt.X(f"{col}:N", sort=None),
                y=alt.Y('count:Q', title='Count'),
                color=alt.Color(f"{col}:N", scale=alt.Scale(scheme='category20'), legend=None),
                tooltip=[alt.Tooltip(f'{col}:N', title='Category'), alt.Tooltip('count:Q', title='Count')]
            ).properties(
                title=title_text
            )

            individual_charts.append(chart)

    if individual_charts:
        combined = alt.hconcat(*individual_charts).resolve_scale(
            x='independent',
            y='independent'
        )
    else:
        combined = None

    return combined



import altair as alt

import altair as alt

import altair as alt

def plot_financial_barcharts(data, categorical_cols, financial_cols, title=None, N=30):
    # Start with an empty horizontal chart
    hconcat_charts = alt.HConcatChart(hconcat=[])

    for financial_col in financial_cols:
        individual_charts = []

        for cat_col in categorical_cols:
            if cat_col == financial_col or cat_col not in data.columns:
                continue

            # Sort the data by financial column and take top N categories
            top_categories_data = data.groupby(cat_col)[financial_col].sum().reset_index().sort_values(financial_col, ascending=False).head(N)
            top_categories = top_categories_data[cat_col].tolist()

            # Add an 'Other' category if there are more than N categories
            if data[cat_col].nunique() > N:
                top_categories.append('Other')
                data.loc[~data[cat_col].isin(top_categories_data[cat_col]), cat_col] = 'Other'

            # Re-aggregate data after filtering to top N categories
            aggregated_data = data.loc[data[cat_col].isin(top_categories)].groupby(cat_col).agg({financial_col: 'sum'}).reset_index()

            # Chart creation logic for either bar or pie chart based on number of categories
            if len(top_categories) <= N:
                chart = alt.Chart(aggregated_data).mark_bar().encode(
                    x=alt.X(f'{cat_col}:N', title=cat_col, sort=None),
                    y=alt.Y(f'{financial_col}:Q', title=f'Sum of {financial_col}'),
                    color=alt.Color(f'{cat_col}:N', legend=None),
                    tooltip=[alt.Tooltip(f'{cat_col}:N'), alt.Tooltip(f'{financial_col}:Q')]
                ).properties(
                    title=f'Sum of {financial_col} by {cat_col}'
                )
            else:
                chart = alt.Chart(aggregated_data).mark_arc().encode(
                    theta=alt.Theta(field=financial_col, type='quantitative'),
                    color=alt.Color(field=cat_col, type='nominal'),
                    tooltip=[alt.Tooltip(field=cat_col, type='nominal'), alt.Tooltip(field=financial_col, type='quantitative')]
                ).properties(
                    title=f'Distribution of {financial_col}'
                )

                # Add legend only for the pie chart
                if len(individual_charts) == 0:
                    legend = alt.Legend(title=cat_col, orient='bottom')
                    chart = chart.configure_legend(legend)

            individual_charts.append(chart)

        # Combine and concatenate the individual charts
        if individual_charts:
            combined_charts_for_col = alt.hconcat(*individual_charts).resolve_scale(color='independent')
            hconcat_charts |= combined_charts_for_col

    if title:
        hconcat_charts = hconcat_charts.properties(title=title)

    return hconcat_charts



import pandas as pd
import altair as alt

def plot_time_series_charts(data, time_date_cols, numerical_cols, title=None):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Initialize an empty vertical chart
    vconcat_charts = alt.VConcatChart(vconcat=[])
    
    for time_col in time_date_cols:
        if time_col not in data.columns:
            continue  # Skip if the column does not exist in the DataFrame
        
        # Convert the time column to datetime if not already
        data[time_col] = pd.to_datetime(data[time_col], errors='coerce')  # 'coerce' will NaT for invalid dates
        data = data.dropna(subset=[time_col])  # Drop rows with NaT (invalid dates)
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