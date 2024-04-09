import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_file
import altair as alt
from visualization.data_visualization import plot_financial_barcharts, plot_categorical_barcharts, plot_distributions_altair, create_scatter_plot, create_scatter_plot_with_line, plot_time_series_charts, visualize_feature_relationships
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
        
        # Check if 'time_date_cols' is defined and has entries
        if len(st.session_state.get('time_date_cols', [])) > 0:
            analysis_subpages.append("Time Series")

        # Define 'analysis_page' before any conditional logic that depends on it
        analysis_page = st.sidebar.radio('Select Analysis Type', analysis_subpages)

        if st.session_state['processed_data'] is not None:
            if analysis_page == "General Analysis":
                # Display general analysis results
                st.write('General analysis results:')
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

            elif analysis_page == "Time Series":
                st.write('Time series analysis results:')
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
                chart_placeholder.altair_chart(chart, use_container_width=True)

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

        if 'normalized_data' in st.session_state and 'processed_data' in st.session_state:
            st.sidebar.header("Prediction Options")
            
            # Task Type Selection
            st.sidebar.header("Task Type Selection")
            task_options = ['Regression', 'Classification']
            selected_task = st.sidebar.radio('Choose the type of task', task_options, key='task_selection')

            # Dynamically set model options based on the selected task type
            if selected_task == 'Regression':
                model_options = ['Linear Regression', 'Random Forest', 'Support Vector Machine']
            elif selected_task == 'Classification':
                model_options = ['Random Forest', 'Support Vector Machine']
            selected_model = st.sidebar.selectbox('Choose a model for prediction', model_options, key='model_selection')
            
            # Introduce Model Parameter UI Elements based on the selected model
            model_params = {}
            if selected_model == 'Random Forest':
                n_estimators = st.sidebar.slider('Number of trees (n_estimators)', 10, 500, 100, 10, key='n_estimators')
                max_depth = st.sidebar.slider('Maximum depth of trees (max_depth)', 1, 32, 5, 1, key='max_depth')
                model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
            elif selected_model == 'Support Vector Machine':
                C = st.sidebar.slider('Penalty parameter C', 0.01, 10.0, 1.0, 0.01, key='C')
                gamma = st.sidebar.selectbox('Kernel coefficient (gamma)', ['scale', 'auto'], index=0, key='gamma')
                model_params = {'C': C, 'gamma': gamma}
                
            # Target Column Selection with Filtering based on Task Type
            if selected_task == 'Classification':
                target_columns = [col for col in st.session_state['processed_data'].columns if st.session_state['processed_data'][col].dtype == 'object']
            else:
                target_columns = st.session_state['normalized_data'].columns.tolist()

            target_column = st.sidebar.selectbox('Select the target column for prediction', target_columns, key='target_column')
            
            # Features Selection
            feature_columns = [col for col in st.session_state['normalized_data'].columns if col != target_column]
            selected_features = st.sidebar.multiselect('Select features to include in the model', feature_columns, default=feature_columns, key='selected_features')

            # Ensure that features are selected before proceeding
            if selected_features:
                # Correct approach to ensure features are numeric and target is categorical for classification
                if selected_task == 'Classification':
                    features_data = st.session_state['normalized_data'][selected_features]
                    target_data = st.session_state['processed_data'][target_column]
                    combined_data = features_data.copy()
                    combined_data[target_column] = target_data.values  # Directly assigning the target column values

                    predictions_df, metrics, shap_summary_df = make_predictions(
                        combined_data, 
                        target_column, 
                        selected_features,
                        selected_model,
                        selected_task,
                        model_params=model_params
                    )
                elif selected_task == 'Regression':
                    # For regression, the approach remains unchanged
                    predictions_df, metrics, shap_summary_df = make_predictions(
                        st.session_state['normalized_data'], 
                        target_column, 
                        selected_features,
                        selected_model,
                        selected_task,
                        model_params=model_params
                    )
                
                # Display predictions and metrics
                st.write(predictions_df)
                for metric, value in metrics.items():
                    st.write(f"{metric}: {value}")

                if 'shap_summary_df' in locals():  # Check if shap_summary_df is defined
                    st.subheader("Feature Importance Based on SHAP Values")

                    # Sort the DataFrame for better visualization
                    shap_summary_df_sorted = shap_summary_df.sort_values(by='SHAP Value', ascending=True)

                    # Create a bar chart
                    plt.figure(figsize=(10, len(shap_summary_df_sorted) / 2))  # Dynamic figure size based on number of features
                    plt.barh(shap_summary_df_sorted.index, shap_summary_df_sorted['SHAP Value'], color='skyblue')
                    plt.xlabel('Mean Absolute SHAP Value')
                    plt.ylabel('Features')
                    plt.title('Feature Importance Based on Mean Absolute SHAP Values')

                    # Display the plot in Streamlit
                    st.pyplot(plt)

            else:
                st.warning("Please select at least one feature to include in the model.")


