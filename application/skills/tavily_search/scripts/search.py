"""
Tavily Search API wrapper for performing internet searches.
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List

def tavily_search(
    query: str,
    api_key: Optional[str] = None,
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
        api_key (str, optional): Your Tavily API key (if not provided, will look for TAVILY_API_KEY env var)
        search_depth (str): "basic" (faster) or "advanced" (more comprehensive)
        include_images (bool): Whether to include image URLs in results
        include_answer (bool): Whether to include an AI-generated answer
        include_raw_content (bool): Whether to include raw content from search results
        max_results (int): Maximum number of results to return
        
    Returns:
        Dict[str, Any]: Search results including URLs, snippets, and optionally images and answer
    """
    # Get API key from parameter or environment variable
    if not api_key:
        api_key = os.environ.get("TAVILY_API_KEY")
    
    if not api_key:
        return {"error": "API key not provided. Either pass api_key parameter or set TAVILY_API_KEY environment variable"}
    
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


def format_search_results(results: Dict[str, Any], max_snippet_length: int = 150) -> str:
    """
    Format search results in a readable way.
    
    Args:
        results (Dict[str, Any]): The search results from tavily_search
        max_snippet_length (int): Maximum length for content snippets
        
    Returns:
        str: Formatted search results as a string
    """
    output = []
    
    # Check for errors
    if "error" in results:
        return f"Error: {results['error']}"
    
    # Add the answer if available
    if "answer" in results and results["answer"]:
        output.append("📝 SUMMARY")
        output.append("-" * 40)
        output.append(results["answer"])
        output.append("")
    
    # Add search results
    if "results" in results and results["results"]:
        output.append("🔍 SEARCH RESULTS")
        output.append("-" * 40)
        
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content")
            
            # Truncate content if needed
            if len(content) > max_snippet_length:
                content = content[:max_snippet_length] + "..."
            
            output.append(f"{i}. {title}")
            output.append(f"   URL: {url}")
            output.append(f"   {content}")
            output.append("")
    
    # Add images if available
    if "images" in results and results["images"]:
        output.append("📸 IMAGES")
        output.append("-" * 40)
        
        for i, image in enumerate(results["images"], 1):
            output.append(f"{i}. {image.get('url', 'No URL')}")
        
        output.append("")
    
    return "\n".join(output)


def summarize_results(results: Dict[str, Any], max_length: int = 500) -> str:
    """
    Create a concise summary from search results.
    
    Args:
        results (Dict[str, Any]): The search results from tavily_search
        max_length (int): Maximum length for the summary
        
    Returns:
        str: A concise summary of the search results
    """
    # If there's already an AI-generated answer, use that
    if "answer" in results and results["answer"]:
        summary = results["answer"]
        if len(summary) <= max_length:
            return summary
        return summary[:max_length] + "..."
    
    # Otherwise, create a summary from the search results
    summary_parts = []
    
    if "results" in results:
        for result in results["results"]:
            title = result.get("title", "")
            content = result.get("content", "")
            
            if title and content:
                summary_parts.append(f"{title}: {content[:100]}...")
    
    summary = " | ".join(summary_parts)
    
    if len(summary) <= max_length:
        return summary
    return summary[:max_length] + "..."


def extract_key_points(results: Dict[str, Any], max_points: int = 5) -> List[str]:
    """
    Extract key points from search results.
    
    Args:
        results (Dict[str, Any]): The search results from tavily_search
        max_points (int): Maximum number of key points to extract
        
    Returns:
        List[str]: A list of key points extracted from the search results
    """
    key_points = []
    
    # If there's an AI-generated answer, extract sentences as key points
    if "answer" in results and results["answer"]:
        sentences = results["answer"].split(". ")
        for sentence in sentences[:max_points]:
            if sentence and not sentence.isspace():
                # Clean up the sentence and add a period if needed
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith((".", "!", "?")):
                    clean_sentence += "."
                key_points.append(clean_sentence)
    
    # If we don't have enough key points yet, extract from search results
    if len(key_points) < max_points and "results" in results:
        for result in results["results"]:
            if len(key_points) >= max_points:
                break
                
            content = result.get("content", "")
            if content:
                sentences = content.split(". ")
                for sentence in sentences:
                    if len(key_points) >= max_points:
                        break
                        
                    if sentence and not sentence.isspace():
                        # Clean up the sentence and add a period if needed
                        clean_sentence = sentence.strip()
                        if not clean_sentence.endswith((".", "!", "?")):
                            clean_sentence += "."
                        
                        # Avoid duplicates
                        if clean_sentence not in key_points:
                            key_points.append(clean_sentence)
    
    return key_points


# Example usage
if __name__ == "__main__":
    # Set your API key here for testing
    api_key = "your_api_key_here"
    
    # Example search
    results = tavily_search(
        query="What is the latest news about artificial intelligence?",
        api_key=api_key
    )
    
    # Format and print results
    formatted_output = format_search_results(results)
    print(formatted_output)
    
    # Extract and print key points
    print("\nKEY POINTS:")
    key_points = extract_key_points(results)
    for i, point in enumerate(key_points, 1):
        print(f"{i}. {point}")