---
name: tavily-search
description: Perform internet searches using the Tavily Search API. Use this skill when the user needs up-to-date information, facts, news, or data that might not be in the model's knowledge. This skill is ideal for answering questions about current events, verifying facts, researching topics, or finding specific information on the web.
---

# Tavily Search Skill

This skill enables internet search capabilities using the Tavily Search API, allowing you to retrieve up-to-date information from the web.

## Usage

Use the search script to perform web searches:

```python
from scripts.search import tavily_search

# Basic search
results = tavily_search(query="your search query")
print(results)

# Advanced search with parameters
results = tavily_search(
    query="your search query",
    search_depth="advanced",  # "basic" (default) or "advanced"
    include_images=False,     # True or False (default)
    include_answer=True,      # True (default) or False
    include_raw_content=False,# True or False (default)
    max_results=5             # Number of results (default: 5)
)
print(results)
```

## Parameters

- `query` (required): The search query string
- `search_depth`: "basic" (faster, less comprehensive) or "advanced" (slower, more comprehensive)
- `include_images`: Whether to include image URLs in results
- `include_answer`: Whether to include an AI-generated answer
- `include_raw_content`: Whether to include raw content from search results
- `max_results`: Maximum number of results to return (default: 5)

## Response Format

The search function returns a dictionary with the following structure:

```python
{
    "results": [
        {
            "url": "https://example.com/page1",
            "content": "Content snippet from the page...",
            "title": "Page Title"
        },
        # More results...
    ],
    "answer": "AI-generated answer based on search results (if include_answer=True)",
    "images": [
        {"url": "https://example.com/image1.jpg"},
        # More images (if include_images=True)...
    ]
}
```

## Best Practices

1. **Be specific**: Use precise search queries for better results
2. **Use advanced search** for complex topics that need deeper research
3. **Include answer** when you need a summarized response
4. **Limit results** to improve processing speed when fewer results are needed
5. **Exclude images** when only text information is needed

## Error Handling

The search function will handle API errors gracefully and return an error message in the results if something goes wrong.