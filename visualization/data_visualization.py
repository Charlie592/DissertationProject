import os
import pandas as pd
import altair as alt
from sklearn.inspection import permutation_importance

def visualize_feature_importances(tpot, X_test, y_test, feature_names, save_dir):    
    best_pipeline = tpot.fitted_pipeline_
    final_estimator = best_pipeline.steps[-1][-1]  # Assume the last step is the model
    
    try:
        # Calculate permutation importances
        result = permutation_importance(final_estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)

        # Create Altair chart
        chart = alt.Chart(importance_df).mark_bar().encode(
            x='Importance:Q',
            y=alt.Y('Feature:N', sort='-x')  # Sort the bars based on the x-axis (Importance)
        ).properties(
            title='Permutation Feature Importance',
            width=800,
            height=400
        )

        # Save chart to file
        chart_path = os.path.join(save_dir, 'permutation_importance_altair.html')
        chart.save(chart_path)

        return chart_path

    except AssertionError as e:
        print(f"Assertion error: {e}")
        return None
    except Exception as e:
        print(f"Error extracting or visualizing feature importances: {e}")
        return None

# Define your X_test, y_test, feature_names, save_dir, and tpot variable properly before calling the function.
# visualize_feature_importances_altair(tpot, X_test, y_test, feature_names, save_dir)

