# Helper Functions for Tavily Search

This document provides detailed information about the helper functions included in the Tavily Search skill.

## Table of Contents
1. [format_search_results](#format_search_results)
2. [summarize_results](#summarize_results)
3. [extract_key_points](#extract_key_points)
4. [Usage Examples](#usage-examples)

## format_search_results

The `format_search_results` function formats search results in a readable way, organizing them into sections.

### Syntax
```python
format_search_results(results: Dict[str, Any], max_snippet_length: int = 150) -> str
```

### Parameters
- `results` (Dict[str, Any]): The search results from tavily_search
- `max_snippet_length` (int, optional): Maximum length for content snippets. Default is 150.

### Return Value
A formatted string containing the search results organized into sections:
- Summary (if available)
- Search Results
- Images (if available)

### Example
```python
from search import tavily_search, format_search_results

results = tavily_search("climate change", api_key="your_api_key")
formatted_output = format_search_results(results)
print(formatted_output)
```

## summarize_results

The `summarize_results` function creates a concise summary from search results.

### Syntax
```python
summarize_results(results: Dict[str, Any], max_length: int = 500) -> str
```

### Parameters
- `results` (Dict[str, Any]): The search results from tavily_search
- `max_length` (int, optional): Maximum length for the summary. Default is 500.

### Return Value
A string containing a concise summary of the search results, limited to the specified maximum length.

### Example
```python
from search import tavily_search, summarize_results

results = tavily_search("renewable energy", api_key="your_api_key")
summary = summarize_results(results)
print(summary)
```

## extract_key_points

The `extract_key_points` function extracts key points from search results.

### Syntax
```python
extract_key_points(results: Dict[str, Any], max_points: int = 5) -> List[str]
```

### Parameters
- `results` (Dict[str, Any]): The search results from tavily_search
- `max_points` (int, optional): Maximum number of key points to extract. Default is 5.

### Return Value
A list of strings, each representing a key point extracted from the search results.

### Example
```python
from search import tavily_search, extract_key_points

results = tavily_search("space exploration", api_key="your_api_key")
key_points = extract_key_points(results)
for i, point in enumerate(key_points, 1):
    print(f"{i}. {point}")
```

## Usage Examples

### Combining Helper Functions

```python
import sys
sys.path.append('skills/tavily_search/scripts')
from search import tavily_search, format_search_results, extract_key_points

# Perform search
results = tavily_search(
    query="latest advancements in quantum computing",
    api_key="your_api_key",
    include_images=True
)

# Print formatted results
print(format_search_results(results))

# Extract and print key points
print("\nKEY POINTS:")
key_points = extract_key_points(results)
for i, point in enumerate(key_points, 1):
    print(f"{i}. {point}")
```

### Creating a Custom Report

```python
import sys
sys.path.append('skills/tavily_search/scripts')
from search import tavily_search, summarize_results, extract_key_points

# Perform search
results = tavily_search(
    query="electric vehicles market trends",
    api_key="your_api_key",
    search_depth="advanced"
)

# Create a custom report
report = []
report.append("# Electric Vehicles Market Report")
report.append("\n## Summary")
report.append(summarize_results(results))

report.append("\n## Key Points")
key_points = extract_key_points(results, max_points=7)
for i, point in enumerate(key_points, 1):
    report.append(f"{i}. {point}")

report.append("\n## Sources")
for i, result in enumerate(results.get("results", []), 1):
    title = result.get("title", "No title")
    url = result.get("url", "No URL")
    report.append(f"{i}. [{title}]({url})")

# Join the report sections
final_report = "\n".join(report)
print(final_report)
```