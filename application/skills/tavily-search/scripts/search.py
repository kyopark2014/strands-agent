"""
Tavily Search API wrapper for performing internet searches.
"""

import os
import json
import requests
from typing import Dict, Any, Optional

def tavily_search(
    query: str,
    search_depth: str = "basic",
    include_images: bool = False,
    include_answer: bool = True,
    include_raw_content: bool = False,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Perform a web search using the Tavily Search API.
    
    Args:
        query (str): The search query
        search_depth (str): "basic" (faster) or "advanced" (more comprehensive)
        include_images (bool): Whether to include image URLs in results
        include_answer (bool): Whether to include an AI-generated answer
        include_raw_content (bool): Whether to include raw content from search results
        max_results (int): Maximum number of results to return
        
    Returns:
        Dict[str, Any]: Search results including URLs, snippets, and optionally images and answer
    """
    # Get API key from environment variable
    api_key = os.environ.get("TAVILY_API_KEY")
    
    if not api_key:
        return {"error": "TAVILY_API_KEY environment variable not found"}
    
    # Tavily API endpoint
    url = "https://api.tavily.com/search"
    
    # Request parameters
    params = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "include_images": include_images,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "max_results": max_results
    }
    
    try:
        # Make the API request
        response = requests.post(url, json=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse and return the JSON response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Failed to parse API response"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Example usage
if __name__ == "__main__":
    results = tavily_search("What is the latest news about artificial intelligence?")
    print(json.dumps(results, indent=2))