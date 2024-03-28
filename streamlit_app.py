import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_file
import altair as alt
from visualization.data_visualization import plot_financial_barcharts, plot_categorical_barcharts, plot_distributions_altair, scatter_plot_with_regression
from models.model_manager import visualize_feature_relationships
from models.predictor import make_predictions
import pandas as pd

# Initialize session state for processed and normalized data
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'show_visualizations' not in st.session_state:
    st.session_state['show_visualizations'] = False

# Main page layout
st.subheader('AI-driven Data Visualization: Revolutionizing Data Analysis through Automation')
uploaded_file = st.file_uploader('Upload your dataset', type=['csv', 'xlsx', 'json', 'txt'])
impute_missing_values = st.checkbox('Impute missing values', key='impute_missing_key')

# Process Data button
if st.button('Process Data'):
    if uploaded_file is not None:
        with st.spinner('Processing... Please wait'):
            # Processing and storing the results in session state
            (
                st.session_state['processed_data'], 
                st.session_state['normalized_data'], 
                st.session_state['labels'],
                financial_cols,  # This should be stored in the session state if you want to preserve it across reruns
                categorical_cols
            ) = process_file(uploaded_file, impute_missing_values=impute_missing_values)
            st.session_state['show_visualizations'] = True
            st.session_state['financial_cols'] = financial_cols  # Add this line to store financial_cols in the session state
            st.session_state['categorical_cols'] = categorical_cols  # Presumably, you want to store this too
            st.success('Data processing complete!')
    else:
        st.error('Please upload a dataset to process.')

# Sidebar navigation and content
if st.session_state['show_visualizations']:
    st.sidebar.header('Visualizations')
    page_options = ["Analysis Results", "Explore Data", "Predictions"]
    page = st.sidebar.selectbox('Select a page', page_options, key='page_selection')

    if page == "Analysis Results":
        # Initialize the subpage list with default page(s)
        analysis_subpages = ["General Analysis", "Anomalies"]

        # Check if 'financial_cols' is defined and has entries
        if len(st.session_state.get('financial_cols', [])) > 0:
            analysis_subpages.append("Financial")

        # Check if 'categorical_cols' is defined and has entries
        if len(st.session_state.get('categorical_cols', [])) > 0:
            analysis_subpages.append("Categorical")

        # Define 'analysis_page' before any conditional logic that depends on it
        analysis_page = st.sidebar.radio('Select Analysis Type', analysis_subpages)

        if st.session_state['processed_data'] is not None:
            if analysis_page == "General Analysis":
                # Display general analysis results
                st.write('General analysis results:')
                if 'processed_data' in st.session_state:
                    processed_data_df = pd.DataFrame(st.session_state['processed_data'])
                    labels = st.session_state['labels']  # Ensure labels are also correctly retrieved or generated
                    
                    figures = visualize_feature_relationships(processed_data_df, labels)
                    for fig in figures:
                        st.pyplot(fig)

            elif analysis_page == "Financial":
                # Display financial analysis results
                st.write('Financial analysis results:')
                # Generate the financial chart using the stored DataFrame and column information
                financial_chart = plot_financial_barcharts(st.session_state['processed_data'], 
                                                        st.session_state.get('categorical_cols', []), 
                                                        st.session_state.get('financial_cols', []))
                st.altair_chart(financial_chart, use_container_width=True)

            elif analysis_page == "Categorical":
                # Display categorical analysis results
                st.write('Categorical analysis results:')
                # Generate the categorical chart using the stored DataFrame and column information
                categorical_chart = plot_categorical_barcharts(st.session_state['processed_data'],
                                                            st.session_state.get('categorical_cols', []))
                st.altair_chart(categorical_chart, use_container_width=True)

            elif analysis_page == "Anomalies":
                # Display anomaly analysis results
                st.write('Anomaly analysis results:')
                # Code for displaying anomaly analysis results goes here
                checkdata = (st.session_state['processed_data'])
                anomalies_chart = plot_distributions_altair(st.session_state['processed_data'], plot_type='boxplot')
                st.altair_chart(anomalies_chart, use_container_width=True)


    elif page == 'Explore Data':
        st.write('Explore Data')
        # Sidebar options for chart selection
        st.sidebar.header("Charts and Plots")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options, key='selected_plot')

        # 'processed_data' has the data to plot
        data = st.session_state['processed_data']
        
        # Only create charts if data is available
        if data is not None and selected_plot:
            column_options = data.columns.tolist()
            
            # Bar plot
            if selected_plot == "Bar plot":
                x_axis = st.sidebar.selectbox("Select category axis", column_options, key='x_axis')
                y_axis = st.sidebar.selectbox("Select value axis", column_options, key='y_axis')
                chart = alt.Chart(data).mark_bar().encode(
                    x=x_axis,
                    y=y_axis
                )
                st.altair_chart(chart, use_container_width=True)  # This line displays the chart

            # Scatter plot with regression line
            elif selected_plot == "Scatter plot":
                # Inside your Streamlit app
                x_axis = st.sidebar.selectbox("Select x-axis", column_options, key='x_axis_scatter')
                y_axis = st.sidebar.selectbox("Select y-axis", column_options, key='y_axis_scatter')

                # Generate the scatter plot with regression line
                scatter_plot = scatter_plot_with_regression(data, x_axis, y_axis)

                # Display the chart
                st.altair_chart(scatter_plot, use_container_width=True)

            # Box plot
            elif selected_plot == "Box plot":
                x_axis = st.sidebar.selectbox("Select category axis", column_options, key='x_axis_box')
                y_axis = st.sidebar.selectbox("Select value axis", column_options, key='y_axis_box')
                chart = alt.Chart(data).mark_boxplot().encode(
                    x=x_axis,
                    y=y_axis
                )
                st.altair_chart(chart, use_container_width=True)  # This line displays the chart

            # Histogram 
            elif selected_plot == "Histogram":
                column = st.sidebar.selectbox("Select a column for histogram", column_options, key='hist_column')
                bins = st.sidebar.slider("Number of bins", min_value=1, max_value=100, value=30, key='hist_bins')
                chart = alt.Chart(data).mark_bar().encode(
                    alt.X(column, bin=alt.Bin(maxbins=bins)),
                    y='count()'
                )
                st.altair_chart(chart, use_container_width=True)

    elif page == 'Predictions':
        st.write('Predictions')

        # Check if normalized_data is available for predictions
        if 'normalized_data' in st.session_state:
            # Move the column selection to the sidebar
            st.sidebar.header("Prediction Options")
            column_to_predict = st.sidebar.selectbox(
                'Select a column to predict', 
                st.session_state['normalized_data'].columns,
                key='column_to_predict'
            )
            
            # Automatically call the prediction function when the column is selected
            predictions_df, metrics = make_predictions(st.session_state['normalized_data'], column_to_predict)
            
            # Display predictions and metrics
            st.write(predictions_df)
            st.write(metrics)


