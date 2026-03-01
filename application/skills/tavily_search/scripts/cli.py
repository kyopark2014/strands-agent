#!/usr/bin/env python3
"""
Command-line interface for Tavily Search.
"""

import argparse
import os
import sys
import json
from search import tavily_search, format_search_results, extract_key_points, summarize_results

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Tavily Search CLI")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--api-key", "-k", help="Tavily API key")
    parser.add_argument("--depth", "-d", choices=["basic", "advanced"], default="basic",
                        help="Search depth (basic or advanced)")
    parser.add_argument("--images", "-i", action="store_true", help="Include images in results")
    parser.add_argument("--no-answer", "-na", action="store_true", help="Exclude AI-generated answer")
    parser.add_argument("--raw", "-r", action="store_true", help="Include raw content in results")
    parser.add_argument("--max-results", "-m", type=int, default=5, help="Maximum number of results")
    parser.add_argument("--format", "-f", choices=["text", "json", "key-points", "summary"],
                        default="text", help="Output format")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Handle query
    if not args.query:
        parser.print_help()
        sys.exit(1)
    query = " ".join(args.query)
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("Error: API key not provided. Use --api-key or set TAVILY_API_KEY environment variable.")
        sys.exit(1)
    
    # Perform search
    results = tavily_search(
        query=query,
        api_key=api_key,
        search_depth=args.depth,
        include_images=args.images,
        include_answer=not args.no_answer,
        include_raw_content=args.raw,
        max_results=args.max_results
    )
    
    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Format output based on selected format
    if args.format == "json":
        output = json.dumps(results, indent=2)
    elif args.format == "key-points":
        key_points = extract_key_points(results)
        output = "\n".join([f"{i}. {point}" for i, point in enumerate(key_points, 1)])
    elif args.format == "summary":
        output = summarize_results(results)
    else:  # text
        output = format_search_results(results)
    
    # Write output to file or stdout
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()