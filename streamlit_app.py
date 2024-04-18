import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_file
import altair as alt
from visualization.data_visualization import plot_financial_barcharts, plot_categorical_barcharts, plot_distributions_altair, create_scatter_plot, create_scatter_plot_with_line, plot_time_series_charts, visualize_feature_relationships, configure_chart
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
                categorical_cols,
                time_date_cols,
                st.session_state['AI_response']
            ) = process_file(uploaded_file, impute_missing_values=impute_missing_values)
            st.session_state['show_visualizations'] = True
            st.session_state['financial_cols'] = financial_cols  # Add this line to store financial_cols in the session state
            st.session_state['categorical_cols'] = categorical_cols  # Presumably, you want to store this too
            st.session_state['time_date_cols'] = time_date_cols  # Store time_date_cols in the session state
            st.success('Data processing complete!')
    else:
        st.error('Please upload a dataset to process.')

# Sidebar navigation and content
if st.session_state['show_visualizations']:
    st.sidebar.header('Visualizations')
    page_options = ["Analysis Results", "Explore Data"]
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
        
        # Check if 'time_date_cols' is defined and has entries
        if len(st.session_state.get('time_date_cols', [])) > 0:
            analysis_subpages.append("Time Series")

        # Define 'analysis_page' before any conditional logic that depends on it
        analysis_page = st.sidebar.radio('Select Analysis Type', analysis_subpages)

        if st.session_state['processed_data'] is not None:
            if analysis_page == "General Analysis":
                # Display general analysis results
                st.write('General analysis results:')
                st.caption('The general analysis section provides an overview of the dataset, highlighting key statistics and distributions. By identifying the numerical columns within the dataset, this section generates histograms and box plots to visualize the data distribution. These visualizations offer insights into the central tendency, spread, and skewness of the data, enabling a deeper understanding of the dataset characteristics. Whether examining revenue figures, customer counts, or any other numerical data, this section provides a comprehensive overview of the dataset dynamics.\n\n')
                st.write("<br>", unsafe_allow_html=True)

                # Code for displaying general analysis results goes here - Taken out to save tokens

                if 'processed_data' in st.session_state:
                    processed_data_df = pd.DataFrame(st.session_state['processed_data'])
                    labels = st.session_state['labels']  # Ensure labels are also correctly retrieved or generated
                    
                    figures, AI_response_fig = visualize_feature_relationships(processed_data_df, labels, st.session_state['AI_response'])
                    for fig in figures:
                        st.pyplot(fig)
                        st.markdown(AI_response_fig[fig], unsafe_allow_html=True)
                        

            elif analysis_page == "Financial":
                # Display financial analysis results
                st.write('Financial analysis results:')
                st.caption('This section is dedicated to showcasing the financial aspects of the dataset. By recognizing all financial-related columns, it presents an in-depth analysis through bar charts that pair these financial metrics with categorical columns. This method of visualization not only simplifies the comparison across different categories but also provides a clear perspective on financial trends and distributions within the dataset. Whether its revenue, expenses, or any other financial parameter, this section brings critical financial insights to the forefront.\n\n')
                st.write("<br>", unsafe_allow_html=True)
                # Generate the financial chart using the stored DataFrame and column information
                financial_chart = plot_financial_barcharts(st.session_state['processed_data'], 
                                                        st.session_state.get('categorical_cols', []), 
                                                        st.session_state.get('financial_cols', []))
                st.altair_chart(financial_chart, use_container_width=True)

            elif analysis_page == "Categorical":
                # Display categorical analysis results
                st.write('Categorical analysis results:')
                st.caption('In the categorical section, our focus shifts to the qualitative aspects of the dataset. Here, all categorical columns are identified and displayed alongside numerical columns in intuitive bar charts. This visualization technique allows for an easy comparison of numerical values across different categories, offering insights into patterns, frequencies, and distributions that might not be immediately evident. Whether analyzing demographic information, product categories, or any other non-numeric data, this section provides a comprehensive overview of categorical data dynamics.\n\n')
                st.write("<br>", unsafe_allow_html=True)
                # Generate the categorical chart using the stored DataFrame and column information
                categorical_chart = plot_categorical_barcharts(st.session_state['processed_data'],
                                                            st.session_state.get('categorical_cols', []))
                st.altair_chart(categorical_chart, use_container_width=True)

            elif analysis_page == "Time Series":
                st.write('Time series analysis results:')
                st.caption('When the dataset includes time or date columns, this section comes into play by offering time series graphs. These visualizations track changes and trends over time, providing a temporal dimension to the data analysis. Time/date charts are invaluable for identifying patterns, seasonal variations, fluctuations, and long-term trends. Whether youre examining sales over the months, website traffic across days, or any time-sensitive data, this section reveals the temporal relationships and dynamics at play.\n\n')
                st.write("<br>", unsafe_allow_html=True)
                processed_data = st.session_state['processed_data']
                
                # Determine numerical columns as those that are not in the categorical or financial columns
                non_time_categorical_cols = set(st.session_state.get('categorical_cols', [])) - set(st.session_state.get('time_date_cols', []))
                numerical_cols = [col for col in processed_data.columns if processed_data[col].dtype in ['int64', 'float64'] and col not in non_time_categorical_cols]
                
                # Generate the time series chart using the stored DataFrame and column information
                time_series_chart = plot_time_series_charts(
                    processed_data,
                    st.session_state.get('time_date_cols', []),
                    numerical_cols
                )
                st.altair_chart(time_series_chart, use_container_width=True)

            elif analysis_page == "Anomalies":
                # Display anomaly analysis results
                st.write('Anomaly analysis results:')
                st.caption('In this section, we delve into the identification of anomalies within the dataset through the use of histograms. Each histogram highlights data distribution for specific variables, with a keen focus on the upper and lower quartiles. Anomalies, or outliers, are visually demarcated, drawing attention to data points that deviate significantly from the norm. This visualization aids in understanding the variability within the data and pinpointing irregularities that may warrant further investigation.\n\n')
                st.write("<br>", unsafe_allow_html=True)
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
                chart = configure_chart(chart)
                st.altair_chart(chart, use_container_width=True)  # This line displays the chart

            # Streamlit app code
            elif selected_plot == "Scatter plot":
                x_axis = st.sidebar.selectbox("Select x-axis", column_options, key='x_axis_scatter')
                y_axis = st.sidebar.selectbox("Select y-axis", column_options, key='y_axis_scatter')

                # Checkbox for showing regression line
                show_regression = st.sidebar.checkbox("Show Regression Line", value=True)

                # Placeholder for the chart
                chart_placeholder = st.empty()

                # Depending on the checkbox, create the appropriate chart
                if show_regression == True:
                    chart = create_scatter_plot_with_line(data, x_axis, y_axis)
                else:
                    chart = create_scatter_plot(data, x_axis, y_axis)

                # Display the chart in the placeholder
                chart = configure_chart(chart)
                chart_placeholder.altair_chart(chart, use_container_width=True)

            # Box plot
            elif selected_plot == "Box plot":
                x_axis = st.sidebar.selectbox("Select category axis", column_options, key='x_axis_box')
                y_axis = st.sidebar.selectbox("Select value axis", column_options, key='y_axis_box')

                # Ensure the y-axis is treated as a quantitative variable
                if data[y_axis].dtype not in ['float64', 'int64']:
                    st.error(f"The selected value axis '{y_axis}' is not continuous.")
                else:
                    chart = alt.Chart(data).mark_boxplot().encode(
                        x=x_axis,
                        y=y_axis
                    )
                    chart = configure_chart(chart)
                    st.altair_chart(chart, use_container_width=True) 
            # Histogram 
            elif selected_plot == "Histogram":
                column = st.sidebar.selectbox("Select a column for histogram", column_options, key='hist_column')
                bins = st.sidebar.slider("Number of bins", min_value=1, max_value=100, value=30, key='hist_bins')
                chart = alt.Chart(data).mark_bar().encode(
                    alt.X(column, bin=alt.Bin(maxbins=bins)),
                    y='count()'
                )
                chart = configure_chart(chart)
                st.altair_chart(chart, use_container_width=True)

            else:
                st.warning("Please select at least one feature to include in the model.")


