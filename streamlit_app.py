import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_file

# Initialize session state for processed and normalized data
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'normalized_data' not in st.session_state:
    st.session_state['normalized_data'] = None
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
        
        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns, key='x_axis_bar')
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns, key='y_axis_bar')
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=46, ha="right")  
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns, key='x_axis_scatter')
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns, key='y_axis_scatter')
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns, key='hist_column')
            bins = st.sidebar.slider("Number of bins", 5, 100, 20, key='hist_bins')
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", data.columns, key='box_column')
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(x=data[column], ax=ax)
            st.pyplot(fig)

    elif page == 'Predictions':
        st.write('Predictions')
        # Display prediction-related widgets or visuals
