"""
Example usage of the Tavily Search API wrapper.
"""

from search import tavily_search
import json

def main():
    print("Tavily Search Example")
    print("---------------------")
    
    # Basic search example
    query = "What are the latest developments in renewable energy?"
    print(f"Performing basic search for: '{query}'")
    
    results = tavily_search(query)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Print the answer if available
    if "answer" in results and results["answer"]:
        print("\nAnswer:")
        print(results["answer"])
    
    # Print search results
    print("\nSearch Results:")
    for i, result in enumerate(results.get("results", []), 1):
        print(f"\n{i}. {result.get('title', 'No title')}")
        print(f"   URL: {result.get('url', 'No URL')}")
        print(f"   Snippet: {result.get('content', 'No content')[:150]}...")

if __name__ == "__main__":
    main()