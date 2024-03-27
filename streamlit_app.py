import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_file
import altair as alt
from visualization.data_visualization import plot_financial_barcharts

# Initialize session state for processed and normalized data
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'show_visualizations' not in st.session_state:
    st.session_state['show_visualizations'] = False

# Main page layout
st.title('AI-driven Data Visualization: Revolutionizing Data Analysis through Automation')
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
                analysis_results, 
                anomaly_distribution_plots, 
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
        st.write('Analysis Results')

        # Initialize the subpage list with default page(s)
        analysis_subpages = ["General Analysis"]

        # Check if 'financial_cols' is defined and has entries
        # This uses .get to avoid KeyError if 'financial_cols' is not in session_state
        if len(st.session_state.get('financial_cols', [])) > 0:
            analysis_subpages.append("Financial")

        # Define 'analysis_page' before any conditional logic that depends on it
        # This allows the sidebar to present the radio buttons for subpage selection
        analysis_page = st.sidebar.radio('Select Analysis Type', analysis_subpages)

        # Check if there is processed data to display
        if st.session_state['processed_data'] is not None:
            if analysis_page == "General Analysis":
                # Display general analysis results
                st.write('General analysis results:')
                # Code for displaying general analysis results goes here
                # For example, you might want to show some metrics or tables
                # st.write(st.session_state['processed_data'].describe())

            elif analysis_page == "Financial":
                # Display financial analysis results
                st.write('Financial analysis results:')
                # Generate the financial chart using the stored DataFrame and column information
                # Ensure that 'categorical_cols' and 'financial_cols' have been set correctly
                chart = plot_financial_barcharts(st.session_state['processed_data'], 
                                                 st.session_state.get('categorical_cols', []), 
                                                 st.session_state.get('financial_cols', []))
                st.altair_chart(chart, use_container_width=True)


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

            # Scatter plot
            elif selected_plot == "Scatter plot":
                x_axis = st.sidebar.selectbox("Select x-axis", column_options, key='x_axis_scatter')
                y_axis = st.sidebar.selectbox("Select y-axis", column_options, key='y_axis_scatter')
                chart = alt.Chart(data).mark_point().encode(
                    x=x_axis,
                    y=y_axis
                )
                st.altair_chart(chart, use_container_width=True)  # This line displays the chart

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
        # Display prediction-related widgets or visuals

