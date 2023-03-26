from pathlib2 import Path

import plotly.graph_objects as go
def plot_performance(y_test, y_pred, model, data_option, pca_option, n_components):
    # Plotting the predicted values with the actual values
    test_indices = [i for i in range(len(y_test))]
    # Create figure and traces
    fig = go.Figure(data=go.Scatter(x=test_indices, y=y_test, name="actual prices", mode='lines', line=dict(width=0.5, color='blue')))
    fig.add_trace(go.Scatter(x=test_indices, y=y_pred, name='predicted prices', mode='lines', line=dict(width=0.5, color='red')))

    # save the plot
    fig.write_html(str(Path.cwd().parent / "figures" / "model_results" /
                    f"{model}_{data_option}_{pca_option}_{n_components}.html"))

    return
