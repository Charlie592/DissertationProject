import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_file
import altair as alt

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
            st.session_state['processed_data'], st.session_state['normalized_data'], analysis_results, anomaly_distribution_plots, _ = process_file(uploaded_file, impute_missing_values=impute_missing_values)
            st.session_state['show_visualizations'] = True
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
        if st.session_state['processed_data'] is not None:
            st.write('Processed data:')
            st.dataframe(st.session_state['processed_data'].head(10))

        if st.session_state['normalized_data'] is not None:
            st.write('Normalized data:')
            st.dataframe(st.session_state['normalized_data'].head(10))

    elif page == 'Explore Data':
        st.write('Explore Data')
        # Sidebar options for chart selection
        st.sidebar.header("Charts and Plots")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options, key='selected_plot')

        # Assuming 'processed_data' has the data to plot
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

            # Histogram - already correct
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

