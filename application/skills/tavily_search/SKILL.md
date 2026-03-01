---
name: tavily_search
description: Perform internet searches using the Tavily Search API. Use this skill when the user needs up-to-date information, facts, news, or data that might not be in the model's knowledge. This skill is ideal for answering questions about current events, verifying facts, researching topics, or finding specific information on the web.
---

# Tavily Search Skill

This skill enables internet search capabilities using the Tavily Search API, allowing you to retrieve up-to-date information from the web.

## Quick Start

```python
# Simple search with default parameters
result = execute_code("""
import sys
sys.path.append('skills/tavily_search/scripts')
from search import tavily_search

# Provide API key directly or set TAVILY_API_KEY environment variable
results = tavily_search("your search query", api_key="your_api_key_here")
print(results["answer"])  # Print the AI-generated answer
""")
```

## Usage Examples

### Basic Search

```python
import sys
sys.path.append('skills/tavily_search/scripts')
from search import tavily_search

# Option 1: Provide API key directly
results = tavily_search(
    query="latest news about artificial intelligence",
    api_key="your_api_key_here"
)

# Option 2: Use environment variable
# import os
# os.environ["TAVILY_API_KEY"] = "your_api_key_here"
# results = tavily_search("latest news about artificial intelligence")

# Print the AI-generated answer
if "answer" in results:
    print(results["answer"])

# Print search results
for i, result in enumerate(results.get("results", []), 1):
    print(f"\n{i}. {result.get('title', 'No title')}")
    print(f"   URL: {result.get('url', 'No URL')}")
    print(f"   Snippet: {result.get('content', 'No content')[:150]}...")
```

### Advanced Search

```python
import sys
sys.path.append('skills/tavily_search/scripts')
from search import tavily_search, format_search_results

# Advanced search with parameters
results = tavily_search(
    query="best restaurants in Seoul",
    api_key="your_api_key_here",
    search_depth="advanced",  # "basic" (default) or "advanced"
    include_images=True,      # True or False (default)
    include_answer=True,      # True (default) or False
    include_raw_content=False,# True or False (default)
    max_results=5             # Number of results (default: 5)
)

# Use the helper function to format results nicely
formatted_output = format_search_results(results)
print(formatted_output)
```

## Parameters

- `query` (required): The search query string
- `api_key` (optional): Your Tavily API key (can also use TAVILY_API_KEY environment variable)
- `search_depth`: "basic" (faster, less comprehensive) or "advanced" (slower, more comprehensive)
- `include_images`: Whether to include image URLs in results
- `include_answer`: Whether to include an AI-generated answer
- `include_raw_content`: Whether to include raw content from search results
- `max_results`: Maximum number of results to return (default: 5)

## Helper Functions

The skill includes helper functions to make working with search results easier:

- `format_search_results()`: Formats search results in a readable way
- `summarize_results()`: Creates a concise summary from search results
- `extract_key_points()`: Extracts key points from search results

For more details on these helper functions, see [references/helpers.md](references/helpers.md).

## Best Practices

1. **Be specific**: Use precise search queries for better results
2. **Use advanced search** for complex topics that need deeper research
3. **Include answer** when you need a summarized response
4. **Limit results** to improve processing speed when fewer results are needed
5. **Format output** using the provided helper functions for better readability

## Error Handling

The search function will handle API errors gracefully and return an error message in the results if something goes wrong. Check for an "error" key in the returned dictionary to handle errors appropriately.