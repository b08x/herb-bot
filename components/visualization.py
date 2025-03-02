"""
Visualization component for the Streamlit application.
"""
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from utils.visualization_utils import (create_document_herb_heatmap,
                                       create_herb_comparison_chart,
                                       create_herb_property_network,
                                       create_term_frequency_chart,
                                       create_timeline_visualization,
                                       extract_herb_mentions)


def initialize_visualization_state():
    """Initialize the visualization state in the session state."""
    if "common_herbs" not in st.session_state:
        # List of common herbs for visualization
        st.session_state.common_herbs = [
            "Chamomile",
            "Echinacea",
            "Ginger",
            "Turmeric",
            "Valerian",
            "Ginkgo",
            "St. John's Wort",
            "Ginseng",
            "Lavender",
            "Peppermint",
            "Ashwagandha",
            "Rhodiola",
            "Holy Basil",
            "Elderberry",
            "Milk Thistle",
            "Dandelion",
            "Nettle",
            "Licorice",
            "Aloe Vera",
            "Calendula",
        ]

    if "herb_properties" not in st.session_state:
        # Sample herb properties for visualization
        st.session_state.herb_properties = [
            {
                "name": "Chamomile",
                "properties": ["Anti-inflammatory", "Calming", "Digestive Aid"],
            },
            {
                "name": "Echinacea",
                "properties": ["Immune Support", "Anti-viral", "Anti-bacterial"],
            },
            {
                "name": "Ginger",
                "properties": ["Anti-inflammatory", "Digestive Aid", "Warming"],
            },
            {
                "name": "Turmeric",
                "properties": ["Anti-inflammatory", "Antioxidant", "Liver Support"],
            },
            {"name": "Valerian", "properties": ["Calming", "Sleep Aid", "Anxiolytic"]},
            {
                "name": "Ashwagandha",
                "properties": ["Adaptogen", "Stress Relief", "Immune Support"],
            },
            {
                "name": "Rhodiola",
                "properties": ["Adaptogen", "Energy", "Cognitive Function"],
            },
        ]

    if "herb_property_values" not in st.session_state:
        # Sample herb property values for radar chart
        st.session_state.herb_property_values = [
            {
                "name": "Ashwagandha",
                "property_values": {
                    "Adaptogenic": 9,
                    "Anti-inflammatory": 6,
                    "Immune Support": 7,
                    "Stress Relief": 9,
                    "Energy": 8,
                    "Sleep Quality": 6,
                },
            },
            {
                "name": "Rhodiola",
                "property_values": {
                    "Adaptogenic": 8,
                    "Anti-inflammatory": 4,
                    "Immune Support": 5,
                    "Stress Relief": 7,
                    "Energy": 9,
                    "Sleep Quality": 3,
                },
            },
            {
                "name": "Holy Basil",
                "property_values": {
                    "Adaptogenic": 7,
                    "Anti-inflammatory": 8,
                    "Immune Support": 6,
                    "Stress Relief": 8,
                    "Energy": 6,
                    "Sleep Quality": 5,
                },
            },
        ]

    if "herb_timeline" not in st.session_state:
        # Sample timeline data
        st.session_state.herb_timeline = [
            {
                "year": 50,
                "herb": "Ginger",
                "event": "Used in Ancient Rome for digestive issues",
            },
            {
                "year": 200,
                "herb": "Echinacea",
                "event": "Used by Native Americans for infections",
            },
            {
                "year": 500,
                "herb": "Turmeric",
                "event": "Documented use in Ayurvedic medicine",
            },
            {
                "year": 800,
                "herb": "Chamomile",
                "event": "Documented medicinal use in Europe",
            },
            {
                "year": 1025,
                "herb": "Ginger",
                "event": "Mentioned in Avicenna's Canon of Medicine",
            },
            {
                "year": 1500,
                "herb": "St. John's Wort",
                "event": "Used for mood disorders in Europe",
            },
            {
                "year": 1700,
                "herb": "Echinacea",
                "event": "Popularized by Eclectic physicians",
            },
            {
                "year": 1850,
                "herb": "Valerian",
                "event": "Used as a sedative in official pharmacopeias",
            },
            {
                "year": 1950,
                "herb": "Ginkgo",
                "event": "Modern research on cognitive benefits begins",
            },
            {
                "year": 1990,
                "herb": "St. John's Wort",
                "event": "Clinical trials for depression published",
            },
            {
                "year": 2000,
                "herb": "Turmeric",
                "event": "Research on curcumin expands significantly",
            },
            {
                "year": 2010,
                "herb": "Ashwagandha",
                "event": "Adaptogen research increases",
            },
        ]


def term_frequency_visualization():
    """Render the term frequency visualization component."""
    st.subheader("Term Frequency Analysis")

    # Text input options
    text_source = st.radio(
        "Text Source", options=["Uploaded Documents", "Custom Text"], horizontal=True
    )

    text_to_analyze = ""

    if text_source == "Uploaded Documents":
        # Get text from uploaded documents
        if (
            "uploaded_documents" in st.session_state
            and st.session_state.uploaded_documents
        ):
            # Create a multiselect for documents
            doc_options = {
                doc_id: doc["metadata"].get("file_name", f"Document {i+1}")
                for i, (doc_id, doc) in enumerate(
                    st.session_state.uploaded_documents.items()
                )
            }

            selected_docs = st.multiselect(
                "Select Documents",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
            )

            if selected_docs:
                # Combine text from selected documents
                for doc_id in selected_docs:
                    doc = st.session_state.uploaded_documents.get(doc_id)
                    if doc:
                        text_to_analyze += doc["text"] + "\n\n"
            else:
                st.info("Please select at least one document.")
        else:
            st.info(
                "No documents uploaded. Please upload documents in the Document Management tab."
            )
    else:
        # Custom text input
        text_to_analyze = st.text_area("Enter text to analyze", height=200)

    # Visualization options
    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider(
            "Number of Terms", min_value=5, max_value=50, value=20, step=5
        )

    with col2:
        # Custom stopwords
        custom_stopwords = st.text_input("Additional stopwords (comma-separated)")

        exclude_words = None
        if custom_stopwords:
            exclude_words = [
                word.strip().lower() for word in custom_stopwords.split(",")
            ]

    # Generate visualization button
    if st.button("Generate Term Frequency Chart", use_container_width=True):
        if text_to_analyze:
            with st.spinner("Generating visualization..."):
                # Create term frequency chart
                fig = create_term_frequency_chart(text_to_analyze, top_n, exclude_words)

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please provide text to analyze.")


def herb_network_visualization():
    """Render the herb network visualization component."""
    st.subheader("Herb-Property Network")

    # Data source options
    data_source = st.radio(
        "Data Source", options=["Sample Data", "Custom Data"], horizontal=True
    )

    herbs_data = []

    if data_source == "Sample Data":
        # Use sample data
        herbs_data = st.session_state.herb_properties
    else:
        # Custom data input
        st.write("Enter herb properties in JSON format:")

        # Example format
        st.code(
            """
[
  {"name": "Herb1", "properties": ["Property1", "Property2"]},
  {"name": "Herb2", "properties": ["Property2", "Property3"]}
]
        """
        )

        custom_data = st.text_area("Herb Properties JSON", height=200)

        if custom_data:
            try:
                herbs_data = json.loads(custom_data)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # Generate visualization button
    if st.button("Generate Network Visualization", use_container_width=True):
        if herbs_data:
            with st.spinner("Generating visualization..."):
                # Create network visualization
                fig = create_herb_property_network(herbs_data)

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please provide herb property data.")


def herb_comparison_visualization():
    """Render the herb comparison visualization component."""
    st.subheader("Herb Property Comparison")

    # Data source options
    data_source = st.radio(
        "Data Source",
        options=["Sample Data", "Custom Data"],
        horizontal=True,
        key="comparison_data_source",
    )

    herbs_data = []
    properties = []

    if data_source == "Sample Data":
        # Use sample data
        herbs_data = st.session_state.herb_property_values

        # Get all properties
        all_properties = set()
        for herb in herbs_data:
            all_properties.update(herb["property_values"].keys())

        properties = sorted(list(all_properties))
    else:
        # Custom data input
        st.write("Enter herb property values in JSON format:")

        # Example format
        st.code(
            """
[
  {
    "name": "Herb1",
    "property_values": {
      "Property1": 8,
      "Property2": 5,
      "Property3": 7
    }
  },
  {
    "name": "Herb2",
    "property_values": {
      "Property1": 6,
      "Property2": 9,
      "Property3": 4
    }
  }
]
        """
        )

        custom_data = st.text_area(
            "Herb Property Values JSON", height=200, key="comparison_json"
        )

        if custom_data:
            try:
                herbs_data = json.loads(custom_data)

                # Get all properties
                all_properties = set()
                for herb in herbs_data:
                    all_properties.update(herb["property_values"].keys())

                properties = sorted(list(all_properties))
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # Generate visualization button
    if st.button("Generate Comparison Chart", use_container_width=True):
        if herbs_data and properties:
            with st.spinner("Generating visualization..."):
                # Create comparison chart
                fig = create_herb_comparison_chart(herbs_data, properties)

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please provide herb property value data.")


def timeline_visualization():
    """Render the timeline visualization component."""
    st.subheader("Historical Timeline")

    # Data source options
    data_source = st.radio(
        "Data Source",
        options=["Sample Data", "Custom Data"],
        horizontal=True,
        key="timeline_data_source",
    )

    events = []

    if data_source == "Sample Data":
        # Use sample data
        events = st.session_state.herb_timeline
    else:
        # Custom data input
        st.write("Enter timeline events in JSON format:")

        # Example format
        st.code(
            """
[
  {"year": 1500, "herb": "Herb1", "event": "Historical event description"},
  {"year": 1700, "herb": "Herb2", "event": "Another historical event"}
]
        """
        )

        custom_data = st.text_area(
            "Timeline Events JSON", height=200, key="timeline_json"
        )

        if custom_data:
            try:
                events = json.loads(custom_data)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # Generate visualization button
    if st.button("Generate Timeline", use_container_width=True):
        if events:
            with st.spinner("Generating visualization..."):
                # Create timeline visualization
                chart = create_timeline_visualization(events)

                # Display the chart
                st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Please provide timeline event data.")


def document_herb_analysis():
    """Render the document herb analysis component."""
    st.subheader("Document Herb Analysis")

    # Check if documents are available
    if (
        "uploaded_documents" not in st.session_state
        or not st.session_state.uploaded_documents
    ):
        st.info(
            "No documents uploaded. Please upload documents in the Document Management tab."
        )
        return

    # Create a multiselect for documents
    doc_options = {
        doc_id: doc["metadata"].get("file_name", f"Document {i+1}")
        for i, (doc_id, doc) in enumerate(st.session_state.uploaded_documents.items())
    }

    selected_docs = st.multiselect(
        "Select Documents",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x],
        key="herb_analysis_docs",
    )

    # Herb selection
    herb_source = st.radio(
        "Herb List Source", options=["Common Herbs", "Custom Herbs"], horizontal=True
    )

    herbs_to_analyze = []

    if herb_source == "Common Herbs":
        # Use common herbs
        herbs_to_analyze = st.multiselect(
            "Select Herbs to Analyze",
            options=st.session_state.common_herbs,
            default=st.session_state.common_herbs[:5],
        )
    else:
        # Custom herbs input
        custom_herbs = st.text_input("Enter herbs to analyze (comma-separated)")

        if custom_herbs:
            herbs_to_analyze = [herb.strip() for herb in custom_herbs.split(",")]

    # Generate visualization button
    if st.button("Generate Herb Analysis", use_container_width=True):
        if selected_docs and herbs_to_analyze:
            with st.spinner("Analyzing documents..."):
                # Create a dictionary of document texts
                documents = {}
                for doc_id in selected_docs:
                    doc = st.session_state.uploaded_documents.get(doc_id)
                    if doc:
                        doc_name = doc["metadata"].get("file_name", doc_id)
                        documents[doc_name] = doc["text"]

                # Create heatmap
                fig = create_document_herb_heatmap(documents, herbs_to_analyze)

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Display detailed mentions
                st.subheader("Detailed Herb Mentions")

                # Create a dataframe for the mentions
                data = []

                for doc_name, doc_text in documents.items():
                    mentions = extract_herb_mentions(doc_text, herbs_to_analyze)

                    for herb, count in mentions.items():
                        data.append(
                            {"Document": doc_name, "Herb": herb, "Mentions": count}
                        )

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No herb mentions found in the selected documents.")
        else:
            st.warning("Please select at least one document and one herb to analyze.")


def render_visualization_tab():
    """Render the visualization tab in the Streamlit application."""
    st.title("Data Visualization")

    # Initialize visualization state
    initialize_visualization_state()

    # Create tabs for different visualizations
    viz_tabs = st.tabs(
        [
            "Term Frequency",
            "Herb Network",
            "Herb Comparison",
            "Timeline",
            "Document Analysis",
        ]
    )

    with viz_tabs[0]:
        term_frequency_visualization()

    with viz_tabs[1]:
        herb_network_visualization()

    with viz_tabs[2]:
        herb_comparison_visualization()

    with viz_tabs[3]:
        timeline_visualization()

    with viz_tabs[4]:
        document_herb_analysis()
