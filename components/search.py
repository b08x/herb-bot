"""
Search interface component for the Streamlit application.
"""
from typing import Any, Dict, List, Optional

import streamlit as st
from services.search_service import search_service


def initialize_search_state():
    """Initialize the search state in the session state."""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    if "search_results" not in st.session_state:
        st.session_state.search_results = []


def perform_search(query: str, num_results: int = 5):
    """
    Perform a search and update the session state.

    Args:
        query: Search query
        num_results: Number of results to return
    """
    # Perform search
    results = search_service.search_with_metadata(query, num_results)

    # Update session state
    st.session_state.search_results = results

    # Add to search history if successful
    if results["success"] and query not in st.session_state.search_history:
        st.session_state.search_history.append(query)

        # Limit history to 10 items
        if len(st.session_state.search_history) > 10:
            st.session_state.search_history.pop(0)


def search_interface():
    """Render the search interface component."""
    st.title("Research Search")

    # Initialize search state
    initialize_search_state()

    # Search input
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("Search for herbalism information")

    with col2:
        num_results = st.selectbox("Results", options=[3, 5, 7, 10], index=1)

    # Search button
    if st.button("Search", use_container_width=True):
        if search_query:
            with st.spinner("Searching..."):
                perform_search(search_query, num_results)
        else:
            st.warning("Please enter a search query.")

    # Display search history
    if st.session_state.search_history:
        with st.expander("Search History"):
            for i, query in enumerate(reversed(st.session_state.search_history)):
                if st.button(query, key=f"history_{i}"):
                    with st.spinner("Searching..."):
                        perform_search(query, num_results)

    # Display search results
    if "search_results" in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results

        if results["success"]:
            st.subheader(f"Search Results for '{results['query']}'")

            if results["num_results"] > 0:
                for i, result in enumerate(results["results"]):
                    with st.container(border=True):
                        st.markdown(f"### [{result['title']}]({result['link']})")
                        st.caption(f"Source: {result['source']}")
                        st.markdown(result["snippet"])

                        # Expand button for more details
                        with st.expander("More Details"):
                            # Display additional metadata if available
                            if "pagemap" in result and result["pagemap"]:
                                if "metatags" in result["pagemap"]:
                                    metatags = result["pagemap"]["metatags"][0]

                                    # Display description if available
                                    if "og:description" in metatags:
                                        st.markdown(
                                            f"**Description:** {metatags['og:description']}"
                                        )

                                    # Display other relevant metatags
                                    if "keywords" in metatags:
                                        st.markdown(
                                            f"**Keywords:** {metatags['keywords']}"
                                        )

                            # Add to chat button
                            if st.button("Add to Chat Context", key=f"add_to_chat_{i}"):
                                if "context_manager" in st.session_state:
                                    # Create a document from the search result
                                    doc_id = f"search_result_{i}"
                                    content = f"# {result['title']}\n\n{result['snippet']}\n\nSource: {result['source']}\nURL: {result['link']}"
                                    metadata = {
                                        "source": result["source"],
                                        "url": result["link"],
                                        "title": result["title"],
                                    }

                                    # Add to context manager
                                    st.session_state.context_manager.document_context.add_document(
                                        doc_id, content, metadata
                                    )

                                    st.success(
                                        f"Added to chat context: {result['title']}"
                                    )
                                else:
                                    st.error("Chat context not available.")
            else:
                st.info(f"No results found for '{results['query']}'")
        else:
            st.error(f"Search failed: {results.get('error', 'Unknown error')}")


def herb_search():
    """Render the specialized herb search component."""
    st.subheader("Herb Search")

    # Herb name input
    herb_name = st.text_input("Herb Name")

    # Search button
    if st.button("Search Herb", use_container_width=True):
        if herb_name:
            with st.spinner(f"Searching for information about {herb_name}..."):
                # Perform specialized herb search
                results = search_service.search_herbs(herb_name)

                # Display results
                if results:
                    for i, result in enumerate(results):
                        with st.container(border=True):
                            st.markdown(f"### [{result['title']}]({result['link']})")
                            st.caption(f"Source: {result['source']}")
                            st.markdown(result["snippet"])
                else:
                    st.info(f"No results found for herb: {herb_name}")
        else:
            st.warning("Please enter an herb name.")


def herb_interaction_search():
    """Render the herb interaction search component."""
    st.subheader("Herb Interaction Search")

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        herb_name = st.text_input("Herb Name", key="interaction_herb")

    with col2:
        substance = st.text_input("Medication/Substance")

    # Search button
    if st.button("Search Interactions", use_container_width=True):
        if herb_name and substance:
            with st.spinner(
                f"Searching for interactions between {herb_name} and {substance}..."
            ):
                # Perform interaction search
                results = search_service.search_herb_interactions(herb_name, substance)

                # Display results
                if results:
                    for i, result in enumerate(results):
                        with st.container(border=True):
                            st.markdown(f"### [{result['title']}]({result['link']})")
                            st.caption(f"Source: {result['source']}")
                            st.markdown(result["snippet"])

                            # Add warning about medical advice
                            if i == 0:
                                st.warning(
                                    "This information is for research purposes only. "
                                    "Always consult a healthcare professional before making "
                                    "decisions about herb-drug interactions."
                                )
                else:
                    st.info(
                        f"No interaction results found for {herb_name} and {substance}"
                    )
        else:
            st.warning("Please enter both an herb name and a medication/substance.")


def fact_check_interface():
    """Render the fact checking interface component."""
    st.subheader("Fact Checking")

    # Statement input
    statement = st.text_area("Enter a statement to fact-check")

    # Fact check button
    if st.button("Fact Check", use_container_width=True):
        if statement:
            with st.spinner("Fact checking..."):
                # Perform fact check
                results = search_service.fact_check(statement)

                # Display results
                if results["num_sources"] > 0:
                    st.write(
                        f"Found {results['num_sources']} sources for fact checking:"
                    )

                    for i, source in enumerate(results["sources"]):
                        with st.container(border=True):
                            st.markdown(f"### [{source['title']}]({source['link']})")
                            st.caption(f"Source: {source['source']}")
                            st.markdown(source["snippet"])
                else:
                    st.info(
                        "No fact-checking sources found. Try rephrasing your statement."
                    )
        else:
            st.warning("Please enter a statement to fact-check.")


def render_search_tab():
    """Render the search tab in the Streamlit application."""
    # Main search interface
    search_interface()

    st.divider()

    # Create two columns for specialized searches
    col1, col2 = st.columns(2)

    with col1:
        # Herb search
        herb_search()

    with col2:
        # Herb interaction search
        herb_interaction_search()

    st.divider()

    # Fact checking interface
    fact_check_interface()
