import json
import math
from typing import Callable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

########################
# Helper functions
########################


def normalize_data(data, method):
    """Normalize data using specified method"""
    if method == "Min-max":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "Z-score":
        return (data - np.mean(data)) / np.std(data)
    return data


def load_jsonl(file):
    """
    Read a text buffer containing JSONL data.
    Each line in the file is a JSON object with "tokens" and "metrics".
    Returns a list of parsed lines.
    """
    lines = []
    for line in file:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        lines.append(data)
    return lines


def color_tokens(tokens, values):
    """
    Create HTML spans with background color for each token based on the corresponding value.
    We'll use a simple min-max scaling for coloring (just for demonstration).
    Mouseover (title attribute) will display the value.
    """
    if not values:
        return " ".join(tokens)

    # Avoid division by zero if all values are the same
    min_val = min(values)
    max_val = max(values)
    scale = max_val - min_val if max_val != min_val else 1.0

    # Build the HTML with inline background color
    colored_text = []
    for token, val in zip(tokens, values):
        # Scale value between 0 and 1
        normalized = (val - min_val) / scale
        # Convert to an RGB shade of red (or use any color mapping you like)
        # Example: from white (255,255,255) to red (255, 100, 100)
        red_intensity = int(255 - (255 - 200) * (1 - normalized))
        green_intensity = int(255 - (255 - 200) * normalized)
        blue_intensity = 200

        color_str = f"rgb({red_intensity},{green_intensity},{blue_intensity})"

        # title attribute for mouseover
        span = f'<span style="background-color: {color_str};" title="{val:.3f}">{token}</span>'
        colored_text.append(span)

    return " ".join(colored_text)


def create_token_plot(
    tokens: list[str],
    metrics: dict[str, list[float]],
    normalization_method: Callable[[list[float]], list[float]] = lambda x: x,
    tokens_per_line: int = 10,
) -> go.Figure:
    """
    Creates a plotly figure displaying tokens in groups (lines) of 'tokens_per_line',
    with each line showing a line plot of the provided metrics over that subset of tokens.
    Each metric has a unique color across all chunks, and the legend shows only one entry
    per metric.

    Args:
        tokens (List[str]): The list of tokens (strings).
        metrics (Dict[str, List[float]]): A dictionary of metric_name -> list of values
                                          (same length as tokens).
        normalization_method (Callable[[List[float]], List[float]]): A function
            that takes a list of float values and returns a list of normalized
            float values (same length).
        tokens_per_line (int, optional): Number of tokens to display per line (subplot).
                                         Defaults to 10.

    Returns:
        go.Figure: A Plotly figure with one row per chunk of tokens.
    """
    # Create a color dictionary for each metric using a Plotly palette (or any palette you like)
    available_colors = px.colors.qualitative.Plotly  # e.g. 10 distinct colors
    metric_names = list(metrics.keys())
    color_dict = {m: available_colors[i % len(available_colors)] for i, m in enumerate(metric_names)}

    # Number of chunks (lines) we'll display
    num_chunks = math.ceil(len(tokens) / tokens_per_line)

    # Create a subplot figure with one column and num_chunks rows
    # The key change here is `shared_yaxes=True`
    fig = make_subplots(rows=num_chunks, cols=1, shared_xaxes=False, shared_yaxes="all", vertical_spacing=0.08)

    # Iterate over chunks
    for chunk_index in range(num_chunks):
        start_i = chunk_index * tokens_per_line
        end_i = start_i + tokens_per_line
        chunk_tokens = tokens[start_i:end_i]

        # For each metric, slice the corresponding chunk of values and optionally normalize
        for metric_name, metric_values in metrics.items():
            chunk_values = metric_values[start_i:end_i]
            normalized_chunk_values = normalization_method(chunk_values)

            # Add the line plot for this metric in the current chunk
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(chunk_tokens))),
                    y=normalized_chunk_values,
                    mode="lines+markers",
                    name=metric_name,
                    legendgroup=metric_name,  # group by metric so only one legend entry
                    showlegend=(chunk_index == 0),  # show legend entry only for the first chunk
                    line=dict(color=color_dict[metric_name]),
                ),
                row=chunk_index + 1,
                col=1,
            )

        # Update the x-axis for this row so tick labels show the actual tokens
        fig.update_xaxes(
            tickmode="array", tickvals=list(range(len(chunk_tokens))), ticktext=chunk_tokens, row=chunk_index + 1, col=1
        )

    # Adjust figure layout
    fig.update_layout(height=300 * num_chunks, showlegend=True, title="Token Metrics Plot")

    return fig


########################
# Streamlit interface
########################

st.title("Rollout Metrics")

# 1. Button to upload a JSONL file
uploaded_file = st.file_uploader("Upload your JSONL file", type=["jsonl"])

# Prepare a space to store the loaded data
if "data" not in st.session_state:
    st.session_state["data"] = []

if uploaded_file is not None:
    # Load the file and store it in session state
    st.session_state["data"] = load_jsonl(uploaded_file)

data = st.session_state["data"]

# If data is loaded, proceed
if data:
    col1, col2 = st.columns(2)

    with col1:
        # 2. Text box (or number input) to choose the index of the line to display
        line_index = st.number_input("Line index to display", min_value=0, max_value=len(data) - 1, value=0)
        # 3. Radio button to choose between text color or line plot
        display_mode = st.radio("Display mode", ["Text Color", "Line Plot"])

    # Retrieve the selected line data
    current_line = data[line_index]
    tokens = current_line["tokens"]
    metrics = current_line["metrics"]  # e.g. {"scoreA": [...], "scoreB": [...], ...}

    if display_mode == "Text Color":
        # 4. If text color is chosen, let the user pick the metric
        metric_list = list(metrics.keys())
        if metric_list:
            with col2:
                selected_metric = st.radio("Choose a metric to color by", metric_list)
            # Retrieve the metric values for the chosen metric
            metric_values = metrics[selected_metric]

            # Generate HTML for colored tokens
            colored_html = color_tokens(tokens, metric_values)
            st.markdown(colored_html, unsafe_allow_html=True)

        else:
            with col2:
                st.warning("No metrics found in this file.")

    else:
        # 5. If line plot is chosen
        #    - Provide checkboxes to select which metrics to plot
        #    - Radio button for normalization method
        with col1:
            normalization_method = st.radio("Normalization", ["No normalization", "Min-max", "Z-score"])
            tokens_per_line = st.number_input("Tokens per line", min_value=3, value=20)

        metric_list = list(metrics.keys())
        selected_metrics = []
        with col2:
            st.write("Metrics to plot:")
            for m in metric_list:
                if st.checkbox(f"{m}", value=True):
                    selected_metrics.append(m)

        fig = create_token_plot(
            tokens=tokens,
            metrics={k: v for k, v in metrics.items() if k in selected_metrics},
            normalization_method=lambda x: normalize_data(x, normalization_method),
            tokens_per_line=tokens_per_line,
        )
        st.plotly_chart(fig)

else:
    st.info("Please upload a JSONL file to proceed.")
