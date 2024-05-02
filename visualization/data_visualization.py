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
import streamlit as st
import json

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import altair as alt
import numpy as np

def plot_distributions_altair(data, plot_type='boxplot', title=None):

    data = data.drop([col for col in data.columns if 'trend' in col], axis=1)
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
    
    # Combine charts into a single visualization, with 1 chart per row
    combined = alt.vconcat(*charts, spacing=30)  # Adding spacing between rows

    if title:
        combined = combined.properties(title=title)

    # Configure the chart with a light theme and add padding
    combined = combined.configure_view(
        stroke=None,  # Removes the border around each view
        continuousWidth=400,  # Adjust width as needed
        continuousHeight=100,  # Adjust height as needed
        strokeWidth=0
    ).configure(
        background='white',
        axis=alt.AxisConfig(
            labelColor='black',
            titleColor='black',
            gridColor='lightgrey',
            domainColor='black',
            tickColor='black',
        ),
        title=alt.TitleConfig(color='black')
    ).configure_view(
        strokeWidth=0,  # Remove the stroke width around the view area
        step=100  # Increase step to allocate more space for each chart
    )

    # Adjust padding around the charts
    combined = combined.properties(padding={"left": 10, "right": 10, "top": 10, "bottom": 10})

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
        title_text = f"Top {actual_categories_count} {col}" if actual_categories_count < N else f"All {col} Categories"

        # If there are more than N categories, include "Other"
        if len(counts) > N:
            top_counts = counts.head(N-1)
            other_count = counts['count'][N-1:].sum()
            top_counts = top_counts.append({col: 'Other', 'count': other_count}, ignore_index=True)
            top_counts = top_counts.sort_values(by='count', ascending=False)
        else:
            top_counts = counts

        if not top_counts.empty:
            chart = alt.Chart(top_counts).mark_bar(cornerRadius=3).encode(
                x=alt.X(f"{col}:N", sort=None),
                y=alt.Y('count:Q', title='Count'),
                color=alt.Color(f"{col}:N", scale=alt.Scale(scheme='category20'), legend=None),
                tooltip=[alt.Tooltip(f'{col}:N', title='Category'), alt.Tooltip('count:Q', title='Count')]
            ).properties(
                width=200,  # Control the width of individual charts
                height=200,  # Control the height of individual charts
                title=title_text
            )
            individual_charts.append(chart)

    # Combine individual charts into rows with 3 charts each
    rows_of_charts = [alt.hconcat(*individual_charts[i:i+3], spacing=20) for i in range(0, len(individual_charts), 3)]

    # Combine rows of charts into a single visualization
    combined = alt.vconcat(*rows_of_charts, spacing=20)

    # Configure the combined chart with a light theme, making the text and lines black
    combined = combined.configure(
        background='white',
        view=alt.ViewConfig(stroke='transparent')
    ).configure_axis(
        labelColor='black',
        titleColor='black',
        gridColor='black',
        domainColor='black',
        tickColor='black'
    ).configure_title(
        color='black'
    ).properties(
        padding={"left": 10, "right": 10, "top": 10, "bottom": 10}  # Add padding around the charts
    ).resolve_scale(
        x='independent', 
        y='independent'
    )

    return combined



def plot_financial_barcharts(data, categorical_cols, financial_cols, title=None, N=30):
    # Create a dictionary to store charts with their respective sum values
    charts_with_sums = {}

    for financial_col in financial_cols:
        for cat_col in categorical_cols:
            if cat_col == financial_col or cat_col not in data.columns:
                continue

            # Aggregate the data
            aggregated_data = (data.groupby(cat_col)[financial_col]
                               .sum()
                               .reset_index()
                               .sort_values(financial_col, ascending=False))

            # Add 'Other' category if needed
            if data[cat_col].nunique() > N:
                aggregated_data = aggregated_data.head(N)
                aggregated_data.loc[aggregated_data.index[-1], cat_col] = 'Other'
                data.loc[~data[cat_col].isin(aggregated_data[cat_col]), cat_col] = 'Other'
                aggregated_data.at[aggregated_data.index[-1], financial_col] = data.loc[data[cat_col] == 'Other', financial_col].sum()

            # Create the chart title
            chart_title = f'Sum of {financial_col} by {cat_col}'

            # Decide between bar or pie chart based on the number of categories
            chart = alt.Chart(aggregated_data)
            if len(aggregated_data) <= N:
                chart = chart.mark_bar().encode(
                    x=alt.X(f'{cat_col}:N', title=cat_col, sort='-y'),
                    y=alt.Y(f'{financial_col}:Q', title=f'Sum of {financial_col}'),
                    color=alt.Color(f'{cat_col}:N', legend=None),
                    tooltip=[alt.Tooltip(f'{cat_col}:N'), alt.Tooltip(f'{financial_col}:Q')]
                )
            else:
                chart = chart.mark_arc().encode(
                    theta=alt.Theta(field=financial_col, type='quantitative'),
                    color=alt.Color(field=cat_col, type='nominal'),
                    tooltip=[alt.Tooltip(field=cat_col, type='nominal'), alt.Tooltip(field=financial_col, type='quantitative')]
                )

            # Store the chart with the total sum
            total_sum = aggregated_data[financial_col].sum()
            charts_with_sums[chart_title] = (total_sum, chart.properties(title=chart_title))

    # Sort charts by their total sum in descending order
    sorted_charts = [chart for _, chart in sorted(charts_with_sums.values(), key=lambda x: x[0], reverse=True)]

    # Group the sorted charts into rows of three
    rows_of_charts = [sorted_charts[i:i + 3] for i in range(0, len(sorted_charts), 3)]

    # Concatenate all the rows of charts
    vconcat_charts = alt.VConcatChart(vconcat=[
        alt.HConcatChart(hconcat=row, spacing=10) for row in rows_of_charts
    ])

    if title:
        vconcat_charts = vconcat_charts.properties(title=title)

    # Configure the final chart's appearance and layout
    vconcat_charts = vconcat_charts.configure(
        background='white',
        view=alt.ViewConfig(stroke='transparent')
    ).configure_axis(
        labelColor='black',
        titleColor='black',
        gridColor='black',
        domainColor='black',
        tickColor='black'
    ).configure_title(
        color='black'
    ).properties(
        padding={"left": 10, "right": 10, "top": 10, "bottom": 10}
    ).resolve_scale(
        x='independent', 
        y='independent'
    )

    return vconcat_charts



import pandas as pd
import altair as alt

import pandas as pd
import altair as alt

def plot_time_series_charts(data, time_date_cols, numerical_cols, title=None, aggregate='daily'):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Initialize an empty list for charts
    charts = []
    
    for time_col in time_date_cols:
        # Convert the time column to datetime if not already and drop NaT values
        data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
        valid_data = data.dropna(subset=[time_col]).copy()  # Copy to avoid SettingWithCopyWarning

        # Set the index to the time_col for resampling
        valid_data.set_index(time_col, inplace=True)
        valid_data.sort_index(inplace=True)

        # Resample if aggregation is specified
        if aggregate:
            if aggregate == 'daily':
                valid_data = valid_data.resample('D').mean().reset_index()
            elif aggregate == 'weekly':
                valid_data = valid_data.resample('W').mean().reset_index()
            elif aggregate == 'monthly':
                valid_data = valid_data.resample('M').mean().reset_index()
            # More resampling options can be added as needed

        # Create line charts for each numerical column
        for num_col in numerical_cols:
            if pd.api.types.is_numeric_dtype(valid_data[num_col]):
                chart = alt.Chart(valid_data).mark_line(point=True).encode(
                    x=alt.X(time_col, title='Date', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y(num_col, title=num_col),
                    tooltip=[alt.Tooltip(time_col, title='Date'), alt.Tooltip(num_col, title=num_col)]
                ).properties(
                    title=f'{num_col} over time' if not title else title,
                    width=300,  # Adjust the width to fit two charts per row
                    height=150  # Adjust the height as needed
                )
                
                charts.append(chart)

    # Combine individual charts into rows with 2 charts each
    rows_of_charts = [alt.hconcat(*charts[i:i+2], spacing=20).resolve_scale(
        x='independent', y='independent'
    ) for i in range(0, len(charts), 2)]

    # Combine rows of charts into a single visualization
    combined_chart = alt.vconcat(*rows_of_charts, spacing=20) if rows_of_charts else alt.value('No time series data available for visualization')
    
    # Configure the combined chart with a light theme and make text and lines black
    combined_chart = combined_chart.configure(
        background='white'
    ).configure_axis(
        labelColor='black',
        titleColor='black',
        gridColor='black',
        domainColor='black',
        tickColor='black'
    ).configure_title(
        color='black'
    ).configure_view(
        stroke='transparent'
    ).properties(
        padding={"left": 10, "right": 30, "top": 10, "bottom": 10}  # Add padding around the charts
    )

    return combined_chart

# Usage example:
# time_series_chart = plot_time_series_charts(data, ['time_column'], ['numerical_column_1', 'numerical_column_2'], aggregate='weekly')


def create_scatter_plot(data, x_col, y_col):
    return alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X(x_col, title=x_col),
        y=alt.Y(y_col, title=y_col),
        tooltip=[x_col, y_col]
        ).properties(
            background='white',
            view=alt.ViewConfig(stroke='transparent')
        ).configure_axis(
            labelColor='black',
            titleColor='black',
            gridColor='black',
            domainColor='black',
            tickColor='black'
        ).configure_title(
            color='black'
        ).properties(
            padding={"left": 10, "right": 10, "top": 10, "bottom": 10}
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
    data_with_trends = data.copy()  # Create a copy of the data
    data_with_trends['trend'] = labels  # Add the trend column to the copied data

    # Example: Calculate the mean for numerical features for each trend
    trend_characteristics = data_with_trends.groupby('trend').mean()

    # Identifying top distinguishing features for one trend as an example
    top_features = trend_characteristics.loc[1].sort_values(ascending=False)[:3].index.tolist()
    #print("Top distinguishing features for trend 1:", top_features)


    # Ensure the figures directory exists if saving figures
    if save_figures:
        import os
        os.makedirs(figures_dir, exist_ok=True)

    AI_response_fig={}
    
    for trend in sorted(data_with_trends['trend'].unique()):
        trend_data = data_with_trends[data_with_trends['trend'] == trend]
        if features:
            trend_data = trend_data[features + ['trend']]  # Select specified features and trend column


        

        # Correlation heatmap
        heatmap_fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(trend_data.drop('trend', axis=1).corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
        plt.title(f"Feature Correlations in Trend {trend+1}")

        AI_response_fig[heatmap_fig]=AI_response[trend]
        if save_figures:
            heatmap_path = f"{figures_dir}/heatmap_trend_{trend}.png"
            heatmap_fig.savefig(heatmap_path)
            figures.append(heatmap_path)
            plt.close(heatmap_fig)
        else:
            figures.append(heatmap_fig)
      
        

    return figures, AI_response_fig


def configure_chart(chart):
    return chart.configure(
        background='white',
        view=alt.ViewConfig(stroke='transparent')
    ).configure_axis(
        labelColor='black',
        titleColor='black',
        gridColor='black',
        domainColor='black',
        tickColor='black'
    ).configure_title(
        color='black'
    ).properties(
        padding={"left": 10, "right": 10, "top": 10, "bottom": 10}
    )


import json
import streamlit as st

def download_chart(chart, filename):
    # Apply the configuration to the chart
    configured_chart = configure_chart(chart)

    # Convert the configured chart to a JSON specification
    chart_spec = json.loads(configured_chart.to_json())

    # HTML template including Vega, Vega-Lite, and Vega-Embed with explicit configuration
    html_template = f"""
    <html>
    <head>
      <!-- Import Vega, Vega-Lite, Vega-Embed -->
      <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    </head>
    <body>
      <div id="vis"></div>
      <script type="text/javascript">
        var spec = {json.dumps(chart_spec)};
        var opt = {{"renderer": "canvas", "actions": true}};
        vegaEmbed('#vis', spec, opt).then(function (result) {{
          console.log(result);
        }}).catch(console.error);
      </script>
    </body>
    </html>
    """

    # Create a download button for the HTML in Streamlit
    st.download_button(
        label="Download chart as HTML",
        data=html_template,
        file_name=f"{filename}.html",
        mime='text/html'
    )


import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from fpdf import FPDF
import os

from bs4 import BeautifulSoup

def strip_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Update your function for saving figures and text to PDF
def save_figures_to_pdf(figures, ai_responses, filename="/Users/charlierobinson/Documents/Code/DissertationCode/DissertationProject/reports/analysis_results.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for index, (fig, response) in enumerate(zip(figures, ai_responses)):
        img_path = f"/Users/charlierobinson/Documents/Code/DissertationCode/DissertationProject/reports/temp_img_{index}.png"  # Unique path for each figure
        fig.savefig(img_path)  # Save current figure
        plt.close(fig)  # Close the figure to free memory after saving

        pdf.image(img_path, x=10, y=None, w=180)  # Add image to PDF at specified location
        
        clean_text = strip_html_tags(response)  # Strip HTML tags if response contains HTML
        pdf.ln(10)  # Add a line break
        pdf.multi_cell(0, 10, clean_text)  # Add text below the image
    
    pdf.output(filename)  # Save the PDF to a file
