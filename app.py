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


def formatted_next_tokens(next_tokens, label_name, val, num_top_tokens=5, new_line_token="\n"):
    base_str = f"{label_name}: {val:.3f}"
    if next_tokens is None:
        return base_str

    top_tokens = [
        (token, prob)
        for token, prob in list(sorted(next_tokens.items(), key=lambda x: x[1], reverse=True))[:num_top_tokens]
    ]
    max_token_len = max(len(token) for token, _ in top_tokens)
    next_tokens_str = new_line_token.join([f"{token:<{max_token_len}} {prob:.3f}" for token, prob in top_tokens])
    return f"{base_str}{new_line_token}----{new_line_token}{next_tokens_str}"


def color_tokens(tokens, values, metric_name, normalization_method, next_tokens=None):
    """
    Create HTML spans with background color for each token based on the corresponding value.
    Includes a custom-styled tooltip for better readability of longer text.

    If the normalization method is "White to Green", the color is white at the minimum value and green at the maximum value.
    If the normalization method is "Red to Green", the color is red at the minimum value and green at the maximum value and white at the mean value.
    """
    if not values:
        return " ".join(tokens)

    # Add CSS for custom tooltip styling
    tooltip_style = """
    <div>
    <style>
    .token-container .token-span {
        position: relative;
        display: inline-block;
    }
    .token-container .token-span:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 5px 10px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        border-radius: 4px;
        font-size: 14px;
        white-space: pre;
        max-width: 1000px;
        z-index: 1000;
        font-family: monospace;
    }
    </style>
    <div class="token-container">
    """

    next_tokens = next_tokens or [None] * len(tokens)
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    max_deviation = max(abs(max_val - mean_val), abs(min_val - mean_val))
    max_abs_val = max(abs(max_val), abs(min_val))

    colored_text = []
    for token, val, next_token in zip(tokens, values, next_tokens):
        # Normalize values according to the method
        if normalization_method == "White to Green":
            # Scale from 0 to 255 for green intensity
            scale = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            red_intensity = 255 - int(255 * scale)  # Fade from 255 to 0
            green_intensity = 255  # Keep green at max
            blue_intensity = 255 - int(255 * scale)  # Fade from 255 to 0
        elif normalization_method == "Red to Green":
            # Scale from -1 to 1 centered at mean
            if max_deviation == 0:
                scale = 0
            else:
                scale = (val - mean_val) / max_deviation

            if scale <= 0:  # Red to White
                red_intensity = 255
                green_intensity = blue_intensity = int(255 * (1 + scale))
            else:  # White to Green
                green_intensity = 255
                red_intensity = blue_intensity = int(255 * (1 - scale))
        else:
            # For unnormalized values: 0 is white, positive is green, negative is red
            # Find the maximum absolute value for scaling
            cap = max_abs_val if max_abs_val > 0 else 1.0

            if val == 0:
                red_intensity = green_intensity = blue_intensity = 255
            elif val > 0:
                # Scale from white to green
                scale = min(1.0, val / cap)  # Scale relative to max value
                green_intensity = 255
                red_intensity = blue_intensity = int(255 * (1 - scale))
            else:  # val < 0
                # Scale from white to red
                scale = min(1.0, abs(val) / cap)  # Scale relative to max value
                red_intensity = 255
                green_intensity = blue_intensity = int(255 * (1 - scale))

        color_str = f"rgb({red_intensity},{green_intensity},{blue_intensity})"

        # Create a more detailed tooltip content
        tooltip_content = f"{val:.3f}" if next_token is None else formatted_next_tokens(next_token, metric_name, val)
        # Escape any quotes in the tooltip content
        tooltip_content = tooltip_content.replace('"', "&quot;")

        span = f'<span class="token-span" style="background-color: {color_str};" data-tooltip="{tooltip_content}">{token}</span>'
        colored_text.append(span)

    return tooltip_style + " ".join(colored_text) + "</div></div>"


def create_token_plot(
    tokens: list[str],
    metrics: dict[str, list[float]],
    next_tokens: list[dict[str, float]] | None = None,
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
    next_tokens = next_tokens or [None] * len(tokens)

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
            chunk_next_tokens = next_tokens[start_i:end_i]

            # Create custom hover text combining token and value

            hover_text = [
                (
                    "<span style='font-family: monospace;'>"
                    + formatted_next_tokens(next_tokens, current_token, raw_value, new_line_token="<br>")
                    + "</span>"
                )
                for current_token, next_tokens, raw_value in zip(chunk_tokens, chunk_next_tokens, chunk_values)
            ]

            # Add the line plot for this metric in the current chunk
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(chunk_tokens))),
                    y=normalized_chunk_values,
                    mode="lines+markers",
                    name=metric_name,
                    legendgroup=metric_name,
                    showlegend=(chunk_index == 0),
                    line=dict(color=color_dict[metric_name]),
                    hovertext=hover_text,
                    hoverinfo="text",  # Show only the custom hover text
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

def html_escape(text):
    # Ensure the input is a string
    s = str(text).strip()
    #s = s.replace("&", "&amp;")
    #s = s.replace("<", "&lt;")
    #s = s.replace(">", "&gt;")
    #s = s.replace('"', "&quot;")
    #s = s.replace("'", "&#39;")
    return s

# Example usage:
if __name__ == "__main__":
    sample = 'Tom & Jerry < "funny" > \'classic\''
    print(html_escape(sample))


st.title("Rollout Metrics")

st.markdown(
    """
    We expect a JSONL file where each line is a JSON object with the following keys:
    - `tokens`: list of tokens (strings)
    - `metrics`: dictionary of metric_name -> list of float values (same length as tokens)
    - `next_tokens` (optional): list of dictionaries (same length as tokens), which each map from a possible next token to its associated probability (or logits)
    """
)

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

    tokens = [html_escape(tok) for tok in tokens]
    next_tokens = [{html_escape(tok):p for tok, p in topk.items()} for topk in current_line.get("next_tokens")]

    if display_mode == "Text Color":
        # 4. If text color is chosen, let the user pick the metric
        metric_list = list(metrics.keys())
        if metric_list:
            with col1:
                normalization_method = st.radio("Color Normalization", ["None", "White to Green", "Red to Green"])
            with col2:
                selected_metric = st.radio("Choose a metric to color by", [f"`{m}`" for m in metric_list])
            # Retrieve the metric values for the chosen metric
            metric_values = metrics[selected_metric.strip("`")]
            # Generate HTML for colored tokens

            colored_html = color_tokens(
                tokens,
                metric_values,
                selected_metric.strip("`"),
                normalization_method=normalization_method,
                next_tokens=next_tokens,
            )
            st.markdown(colored_html, unsafe_allow_html=True)

        else:
            with col2:
                st.warning("No metrics found in this file.")

    else:
        # 5. If line plot is chosen
        #    - Provide checkboxes to select which metrics to plot
        #    - Radio button for normalization method
        with col1:
            normalization_method = st.radio("Normalization", ["None", "Min-max", "Z-score"])
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
            next_tokens=next_tokens,
            tokens_per_line=tokens_per_line,
        )
        st.plotly_chart(fig)

else:
    st.info("Please upload a JSONL file to proceed.")
