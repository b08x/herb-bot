"""
Service for interacting with Google Custom Search API.
"""
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API credentials
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Flag to track if Google API is available
GOOGLE_API_AVAILABLE = False

try:
    from googleapiclient.discovery import build

    GOOGLE_API_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Google API client not available: {e}")
    print("Search functionality will be limited.")


class SearchService:
    """Service for interacting with Google Custom Search API."""

    def __init__(self):
        """Initialize the search service."""
        self.service = None
        self.search_engine_id = GOOGLE_SEARCH_ENGINE_ID
        self.api_available = GOOGLE_API_AVAILABLE

        if self.api_available:
            try:
                self.service = build(
                    "customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY
                )
            except Exception as e:
                print(f"Error initializing Google Search API: {e}")
                self.api_available = False

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a search using Google Custom Search.

        Args:
            query: Search query
            num_results: Number of results to return (max 10)

        Returns:
            List of search result dictionaries
        """
        # Check if API is available
        if not self.api_available or self.service is None:
            print("Google Search API is not available")
            return [
                {
                    "title": "API Not Available",
                    "link": "#",
                    "snippet": "The Google Search API is not available. Please check your API keys and dependencies.",
                    "source": "System",
                    "pagemap": {},
                }
            ]

        try:
            # Ensure num_results is within valid range (1-10)
            num_results = max(1, min(10, num_results))

            # Execute search
            result = (
                self.service.cse()
                .list(q=query, cx=self.search_engine_id, num=num_results)
                .execute()
            )

            # Extract and format results
            search_results = []

            if "items" in result:
                for item in result["items"]:
                    search_results.append(
                        {
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "source": item.get("displayLink", ""),
                            "pagemap": item.get("pagemap", {}),
                        }
                    )

            return search_results

        except Exception as e:
            print(f"Error performing search: {e}")
            return [
                {
                    "title": "Search Error",
                    "link": "#",
                    "snippet": f"Error performing search: {str(e)}",
                    "source": "System",
                    "pagemap": {},
                }
            ]

    def search_with_metadata(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Perform a search and include metadata about the search.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        try:
            results = self.search(query, num_results)

            return {
                "query": query,
                "num_results": len(results),
                "results": results,
                "success": True,
            }

        except Exception as e:
            print(f"Error performing search: {e}")
            return {
                "query": query,
                "num_results": 0,
                "results": [],
                "success": False,
                "error": str(e),
            }

    def fact_check(self, statement: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Perform a fact-checking search for a statement.

        Args:
            statement: Statement to fact-check
            num_results: Number of results to return

        Returns:
            Dictionary with fact-checking results
        """
        # Create a fact-checking query
        query = f"fact check {statement}"

        # Perform search
        results = self.search(query, num_results)

        # Format response
        return {"statement": statement, "sources": results, "num_sources": len(results)}

    def search_herbs(
        self, herb_name: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform a search specifically for herb information.

        Args:
            herb_name: Name of the herb to search for
            num_results: Number of results to return

        Returns:
            List of search result dictionaries
        """
        # Create a specialized query for herbs
        query = f"{herb_name} herb medicinal properties research"

        # Perform search
        return self.search(query, num_results)

    def search_herb_interactions(
        self, herb_name: str, substance: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for interactions between an herb and another substance.

        Args:
            herb_name: Name of the herb
            substance: Name of the substance (drug, herb, etc.)
            num_results: Number of results to return

        Returns:
            List of search result dictionaries
        """
        # Create a specialized query for interactions
        query = f"{herb_name} interaction with {substance} research"

        # Perform search
        return self.search(query, num_results)


# Create a singleton instance
search_service = SearchService()
