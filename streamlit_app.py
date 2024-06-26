import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_file
import altair as alt
from visualization.data_visualization import plot_financial_barcharts, plot_categorical_barcharts, plot_distributions_altair, create_scatter_plot, create_scatter_plot_with_line, plot_time_series_charts, visualize_feature_relationships, configure_chart, download_chart, save_figures_to_pdf
import pandas as pd

# Initialize session state for processed and normalized data
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'show_visualizations' not in st.session_state:
    st.session_state['show_visualizations'] = False

# Main page layout
st.subheader('AI-driven Data Visualization: Revolutionizing Data Analysis through Automation')
uploaded_file = st.file_uploader('Upload your dataset', type=['csv', 'xlsx', 'json', 'txt'], help="""Upload a dataset in CSV, XLSX, JSON, or TXT format to begin the analysis process. 
                                 The uploaded dataset will be processed to generate visualizations and insights that aid in understanding the data dynamics.""")
impute_missing_values = st.checkbox('Impute missing values', key='impute_missing_key', help="""Impute missing values by replacing them with the mean, median, or mode of the respective column. 
                                    This step ensures that the dataset is complete and ready for analysis. Missing values can skew the results and hinder the accuracy of the analysis. 
                                    By imputing these values, we ensure that the dataset is robust and reliable for further processing.""")

# Process Data button
if st.button('Process Data'):
    if uploaded_file is not None:
        with st.spinner('Processing... Please wait'):
            # Processing and storing the results in session state
            (
                st.session_state['processed_data'], 
                st.session_state['normalized_data'], 
                st.session_state['labels'],
                financial_cols,  
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
                st.caption("""Explore the foundational overview of your dataset in the General Analysis section. 
                           Here, we highlight key statistics and distribution patterns through sophisticated visualizations like correlation heatmaps. 
                           Discover trends, understand data characteristics, and gain actionable insights with AI-enhanced summaries tailored to highlight the most relevant data points.\n\n""")
                st.write("<br>", unsafe_allow_html=True)

                # Code for displaying general analysis results goes here - Taken out to save tokens

                if 'processed_data' in st.session_state:
                    processed_data_df = pd.DataFrame(st.session_state['processed_data'])
                    labels = st.session_state['labels']  # Ensure labels are also correctly retrieved or generated
                    figures, AI_response_fig = visualize_feature_relationships(processed_data_df, labels, st.session_state['AI_response'])
                    ai_responses = []
                    for fig in figures:
                        st.pyplot(fig)  # Display each figure
                        response_text = AI_response_fig[fig]
                        st.markdown(response_text, unsafe_allow_html=True)
                        ai_responses.append(response_text)
                        plt.close(fig)

                    if st.button("Generate PDF Report"):
                        ai_responses = [AI_response_fig[fig] for fig in figures]
                        pdf_filename = "/Users/charlierobinson/Documents/Code/DissertationCode/DissertationProject/reports/analysis_results.pdf"
                        save_figures_to_pdf(figures, ai_responses, pdf_filename)
                        with open(pdf_filename, "rb") as f:
                            st.download_button("Download now", f, "analysis_results.pdf", "application/pdf")

            elif analysis_page == "Financial":
                # Display financial analysis results
                st.write('Financial analysis results:')
                st.caption("""This section is dedicated to showcasing the financial aspects of the dataset. By recognizing all financial-related columns, 
                           it presents an in-depth analysis through bar charts that pair these financial metrics with categorical columns. 
                           This method of visualization not only simplifies the comparison across different categories but also provides a clear perspective on financial trends and distributions within the dataset. 
                           Whether its revenue, expenses, or any other financial parameter, this section brings critical financial insights to the forefront.\n\n""")
                st.write("<br>", unsafe_allow_html=True)
                # Generate the financial chart using the stored DataFrame and column information
                financial_chart = plot_financial_barcharts(st.session_state['processed_data'], 
                                                        st.session_state.get('categorical_cols', []), 
                                                        st.session_state.get('financial_cols', []))
                financial_chart = financial_chart.properties(
                    autosize={'type': 'fit', 'contains': 'padding'}) 
                st.altair_chart(financial_chart, use_container_width=True)
                download_chart(financial_chart, "financial_chart")

            elif analysis_page == "Categorical":
                # Display categorical analysis results
                st.write('Categorical analysis results:')
                st.caption("""In the categorical section, our focus shifts to the qualitative aspects of the dataset. 
                           Here, all categorical columns are identified and displayed alongside numerical columns in intuitive bar charts. 
                           This visualization technique allows for an easy comparison of numerical values across different categories, offering insights into patterns, frequencies, 
                           and distributions that might not be immediately evident. Whether analyzing demographic information, product categories, or any other non-numeric data, 
                           this section provides a comprehensive overview of categorical data dynamics.\n\n""")
                st.write("<br>", unsafe_allow_html=True)
                # Generate the categorical chart using the stored DataFrame and column information
                categorical_chart = plot_categorical_barcharts(st.session_state['processed_data'],
                                                            st.session_state.get('categorical_cols', []))
                categorical_chart = categorical_chart.properties(
                    autosize={'type': 'fit', 'contains': 'padding'})
                st.altair_chart(categorical_chart, use_container_width=True)
                download_chart(categorical_chart, "categorical_chart")

            elif analysis_page == "Time Series":
                st.write('Time series analysis results:')
                st.caption("""When the dataset includes time or date columns, this section comes into play by offering time series graphs. 
                           These visualizations track changes and trends over time, providing a temporal dimension to the data analysis. 
                           Time/date charts are invaluable for identifying patterns, seasonal variations, fluctuations, and long-term trends. 
                           Whether youre examining sales over the months, website traffic across days, or any time-sensitive data, this section reveals the temporal relationships and dynamics at play.\n\n""")
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
                time_series_chart = time_series_chart.properties(
                    autosize={'type': 'fit', 'contains': 'padding'}) 
                st.altair_chart(time_series_chart, use_container_width=True)
                download_chart(time_series_chart, "time_series_chart")

            elif analysis_page == "Anomalies":
                # Display anomaly analysis results
                st.write('Anomaly analysis results:')
                st.caption("""In this section, we delve into the identification of anomalies within the dataset through the use of histograms. 
                           Each histogram highlights data distribution for specific variables, with a keen focus on the upper and lower quartiles. 
                           Anomalies, or outliers, are visually demarcated, drawing attention to data points that deviate significantly from the norm. 
                           This visualization aids in understanding the variability within the data and pinpointing irregularities that may warrant further investigation.\n\n""")
                st.write("<br>", unsafe_allow_html=True)
                # Code for displaying anomaly analysis results goes here
                checkdata = (st.session_state['processed_data'])
                anomalies_chart = plot_distributions_altair(st.session_state['processed_data'], plot_type='boxplot')
                st.altair_chart(anomalies_chart, use_container_width=True)
                download_chart(anomalies_chart, "anomalies_chart")


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
        
            if selected_plot == "Bar plot":
                x_axis = st.sidebar.selectbox("Select category axis", column_options, key='x_axis')
                y_axis = st.sidebar.selectbox("Select value axis", column_options, key='y_axis')
                chart = alt.Chart(data).mark_bar().encode(
                    x=x_axis,
                    y=y_axis
                ).properties(
                    width=600,
                    height=300,
                    background='white',
                    padding={"left": 10, "right": 10, "top": 10, "bottom": 10}
                ).configure_view(
                    stroke='transparent'
                ).configure_axis(
                    labelColor='black',
                    titleColor='black',
                    gridColor='black',
                    domainColor='black',
                    tickColor='black'
                ).configure_title(
                    color='black'
                )
                
                chart = configure_chart(chart)
                st.altair_chart(chart, use_container_width=True) 
                download_chart(chart, "bar_plot")

            elif selected_plot == "Scatter plot":
                x_axis = st.sidebar.selectbox("Select x-axis", column_options, key='x_axis_scatter')
                y_axis = st.sidebar.selectbox("Select y-axis", column_options, key='y_axis_scatter')
                show_regression = st.sidebar.checkbox("Show Regression Line", value=False)
                if show_regression:
                    chart = create_scatter_plot_with_line(data, x_axis, y_axis)
                else:
                    chart = create_scatter_plot(data, x_axis, y_axis)
                chart = configure_chart(chart)
                st.altair_chart(chart, use_container_width=True)
                download_chart(chart, "scatter_plot")

            elif selected_plot == "Histogram":
                column = st.sidebar.selectbox("Select a column for histogram", column_options, key='hist_column')
                bins = st.sidebar.slider("Number of bins", min_value=1, max_value=100, value=30, key='hist_bins')
                chart = alt.Chart(data).mark_bar().encode(
                    alt.X(column, bin=alt.Bin(maxbins=bins)),
                    y='count()'
                ).properties(
                    width=600,
                    height=300,
                    background='white',
                    padding={"left": 10, "right": 10, "top": 10, "bottom": 10}
                ).configure_view(
                    stroke='transparent'
                ).configure_axis(
                    labelColor='black',
                    titleColor='black',
                    gridColor='black',
                    domainColor='black',
                    tickColor='black'
                ).configure_title(
                    color='black'
                )
                chart = configure_chart(chart)
                st.altair_chart(chart, use_container_width=True)
                download_chart(chart, "histogram")

            elif selected_plot == "Box plot":
                x_axis = st.sidebar.selectbox("Select category axis", column_options, key='x_axis_box')
                y_axis = st.sidebar.selectbox("Select value axis", column_options, key='y_axis_box')

                # Ensure the y-axis is treated as a quantitative variable
                if data[y_axis].dtype not in ['float64', 'int64']:
                    st.warning(f"The selected value axis '{y_axis}' is not continuous.")
                else:
                    chart = alt.Chart(data).mark_boxplot().encode(
                        x=x_axis, 
                        y=y_axis
                ).properties(
                    width=600,
                    height=300,
                    background='white',
                    padding={"left": 10, "right": 10, "top": 10, "bottom": 10}
                ).configure_view(
                    stroke='transparent'
                ).configure_axis(
                    labelColor='black',
                    titleColor='black',
                    gridColor='black',
                    domainColor='black',
                    tickColor='black'
                ).configure_title(
                    color='black'
                )
                    st.altair_chart(chart, use_container_width=True)
                    download_chart(chart, "box_plot")



            else:
                st.warning("Please select at least one feature to include in the model.")


