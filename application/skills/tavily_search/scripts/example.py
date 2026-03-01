"""
Example usage of the Tavily Search API wrapper.
"""

import os
import sys
from search import tavily_search, format_search_results, extract_key_points

def run_example_search():
    """Run an example search and display the results."""
    print("Tavily Search Example")
    print("=" * 50)
    
    # Get API key from environment or use a placeholder
    api_key = os.environ.get("TAVILY_API_KEY", "your_api_key_here")
    
    # Ask for query
    query = input("Enter your search query: ")
    if not query:
        query = "What are the latest developments in renewable energy?"
        print(f"Using default query: '{query}'")
    
    print(f"\nSearching for: '{query}'")
    print("=" * 50)
    
    # Perform search
    results = tavily_search(
        query=query,
        api_key=api_key,
        include_images=True
    )
    
    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
        if "API key" in results["error"]:
            print("\nTo use this example, you need to:")
            print("1. Set your Tavily API key as an environment variable:")
            print("   export TAVILY_API_KEY=your_api_key_here")
            print("2. Or modify this script to include your API key directly.")
        return
    
    # Format and print results
    formatted_output = format_search_results(results)
    print(formatted_output)
    
    # Extract and print key points
    print("KEY POINTS:")
    print("=" * 50)
    key_points = extract_key_points(results)
    for i, point in enumerate(key_points, 1):
        print(f"{i}. {point}")

if __name__ == "__main__":
    run_example_search()