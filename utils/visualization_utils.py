"""
Utility functions for data visualization.
"""
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_term_frequency_chart(
    text: str, top_n: int = 20, exclude_words: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a bar chart of term frequencies from text.

    Args:
        text: Text to analyze
        top_n: Number of top terms to include
        exclude_words: List of words to exclude (stopwords)

    Returns:
        Plotly figure object
    """
    if exclude_words is None:
        exclude_words = [
            "the",
            "and",
            "to",
            "of",
            "a",
            "in",
            "for",
            "is",
            "on",
            "that",
            "by",
            "this",
            "with",
            "i",
            "you",
            "it",
            "not",
            "or",
            "be",
            "are",
            "from",
            "at",
            "as",
            "your",
            "have",
            "more",
            "an",
            "was",
            "we",
            "will",
            "can",
            "all",
            "has",
            "but",
            "our",
            "one",
            "other",
            "do",
            "they",
            "which",
            "their",
        ]

    # Convert to lowercase and split into words
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter out excluded words and single characters
    filtered_words = [
        word for word in words if word not in exclude_words and len(word) > 1
    ]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Get top N words
    top_words = word_counts.most_common(top_n)

    # Create dataframe
    df = pd.DataFrame(top_words, columns=["Term", "Frequency"])

    # Create bar chart
    fig = px.bar(
        df,
        x="Frequency",
        y="Term",
        orientation="h",
        title=f"Top {top_n} Terms by Frequency",
        labels={"Frequency": "Occurrence Count", "Term": ""},
        height=500,
    )

    # Update layout
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def create_herb_property_network(herbs_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a network visualization of herbs and their properties.

    Args:
        herbs_data: List of dictionaries with herb information
            Each dict should have 'name' and 'properties' keys

    Returns:
        Plotly figure object
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes and edges
    for herb in herbs_data:
        herb_name = herb["name"]
        G.add_node(herb_name, type="herb")

        for prop in herb["properties"]:
            # Add property node if it doesn't exist
            if not G.has_node(prop):
                G.add_node(prop, type="property")

            # Add edge between herb and property
            G.add_edge(herb_name, prop)

    # Use networkx spring layout
    pos = nx.spring_layout(G, seed=42)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node traces (separate for herbs and properties)
    herb_nodes_x = []
    herb_nodes_y = []
    herb_nodes_text = []

    property_nodes_x = []
    property_nodes_y = []
    property_nodes_text = []

    for node in G.nodes():
        x, y = pos[node]
        if G.nodes[node]["type"] == "herb":
            herb_nodes_x.append(x)
            herb_nodes_y.append(y)
            herb_nodes_text.append(node)
        else:
            property_nodes_x.append(x)
            property_nodes_y.append(y)
            property_nodes_text.append(node)

    herb_node_trace = go.Scatter(
        x=herb_nodes_x,
        y=herb_nodes_y,
        mode="markers",
        hoverinfo="text",
        text=herb_nodes_text,
        marker=dict(color="#6175c1", size=15, line=dict(width=1, color="#000")),
        name="Herbs",
    )

    property_node_trace = go.Scatter(
        x=property_nodes_x,
        y=property_nodes_y,
        mode="markers",
        hoverinfo="text",
        text=property_nodes_text,
        marker=dict(
            color="#84ba5b", size=10, symbol="diamond", line=dict(width=1, color="#000")
        ),
        name="Properties",
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, herb_node_trace, property_node_trace],
        layout=go.Layout(
            title="Herb-Property Network",
            titlefont=dict(size=16),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


def create_herb_comparison_chart(
    herbs_data: List[Dict[str, Any]], properties: List[str]
) -> go.Figure:
    """
    Create a radar chart comparing multiple herbs across properties.

    Args:
        herbs_data: List of dictionaries with herb information
            Each dict should have 'name' and 'property_values' keys
        properties: List of properties to compare

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    for herb in herbs_data:
        values = [herb["property_values"].get(prop, 0) for prop in properties]

        # Add a value at the end that matches the first value to close the polygon
        values.append(values[0])
        properties_closed = properties + [properties[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values, theta=properties_closed, fill="toself", name=herb["name"]
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title="Herb Properties Comparison",
        showlegend=True,
    )

    return fig


def create_timeline_visualization(events: List[Dict[str, Any]]) -> alt.Chart:
    """
    Create a timeline visualization of historical herb usage.

    Args:
        events: List of dictionaries with historical events
            Each dict should have 'year', 'event', and 'herb' keys

    Returns:
        Altair chart object
    """
    # Convert to DataFrame
    df = pd.DataFrame(events)

    # Sort by year
    df = df.sort_values("year")

    # Create Altair chart
    chart = (
        alt.Chart(df)
        .mark_circle(size=100)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("herb:N", title="Herb"),
            tooltip=["year", "herb", "event"],
            color=alt.Color("herb:N", legend=None),
        )
        .properties(width=800, height=400, title="Historical Timeline of Herb Usage")
    )

    # Add text labels
    text = chart.mark_text(align="left", baseline="middle", dx=7, fontSize=12).encode(
        text="event:N"
    )

    return (chart + text).interactive()


def extract_herb_mentions(text: str, herb_list: List[str]) -> Dict[str, int]:
    """
    Extract and count herb mentions from text.

    Args:
        text: Text to analyze
        herb_list: List of herbs to look for

    Returns:
        Dictionary mapping herbs to mention counts
    """
    mentions = {}

    for herb in herb_list:
        # Create regex pattern for the herb (word boundary to match whole words)
        pattern = r"\b" + re.escape(herb) + r"\b"

        # Count occurrences
        count = len(re.findall(pattern, text, re.IGNORECASE))

        if count > 0:
            mentions[herb] = count

    return mentions


def create_document_herb_heatmap(
    documents: Dict[str, str], herb_list: List[str]
) -> go.Figure:
    """
    Create a heatmap of herb mentions across documents.

    Args:
        documents: Dictionary mapping document names to text content
        herb_list: List of herbs to look for

    Returns:
        Plotly figure object
    """
    # Extract herb mentions for each document
    data = []

    for doc_name, doc_text in documents.items():
        mentions = extract_herb_mentions(doc_text, herb_list)

        for herb, count in mentions.items():
            data.append({"Document": doc_name, "Herb": herb, "Mentions": count})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Check if dataframe is empty
    if df.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No herb mentions found in the selected documents",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title="Herb Mentions Across Documents",
            height=500,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    # Pivot for heatmap format
    pivot_df = df.pivot(index="Document", columns="Herb", values="Mentions").fillna(0)

    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Herb", y="Document", color="Mentions"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="Viridis",
        title="Herb Mentions Across Documents",
    )

    fig.update_layout(
        xaxis={"side": "top"}, height=500, margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig
