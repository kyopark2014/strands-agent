---
name: retrieve
description: Search a knowledge base using Amazon Bedrock RAG (Retrieval-Augmented Generation). Use when users want to find information from uploaded documents, ask questions about stored knowledge, or retrieve relevant content from the knowledge base. Returns matched content with source references and URLs.
---

# Knowledge Base Retrieve

Search and retrieve relevant documents from an Amazon Bedrock Knowledge Base using RAG.

## Quick Start

Use the retrieve script to query the knowledge base by keyword:

```python
import subprocess
result = subprocess.run(['python', 'scripts/retrieve_search.py', 'keyword'], 
                       capture_output=True, text=True, cwd='retrieve')
print(result.stdout)
```

## Script Location

The retrieve script is located at `skills/retrieve/scripts/retrieve_search.py` relative to the application working directory.
**IMPORTANT**: Always use the FULL path `skills/retrieve/scripts/retrieve_search.py` — do NOT shorten to `scripts/retrieve_search.py`.

## Features

- **Knowledge Base Search**: Query documents indexed in Amazon Bedrock Knowledge Base
- **RAG-based Retrieval**: Uses vector search to find the most relevant content
- **Source References**: Returns source URLs and document titles for each result
- **Auto Recovery**: Automatically updates knowledge base ID if the resource is not found
- **S3 & Web Sources**: Supports both S3 document and web-crawled content locations
- **Top Results**: Returns up to 5 most relevant document chunks

## Usage Examples

### Basic Query
```python
# Search for information about a topic
result = subprocess.run(['python', 'scripts/retrieve_search.py', '클라우드 아키텍처'], 
                       capture_output=True, text=True, cwd='retrieve')
```

### Technical Query
```python
# Search for technical documentation
result = subprocess.run(['python', 'scripts/retrieve_search.py', 'API authentication'], 
                       capture_output=True, text=True, cwd='retrieve')
```

### Concept Query
```python
# Search for explanations
result = subprocess.run(['python', 'scripts/retrieve_search.py', '보안 정책'], 
                       capture_output=True, text=True, cwd='retrieve')
```

## Output Format

The script returns a JSON array. Each element contains:

```json
[
  {
    "contents": "matched text from the knowledge base",
    "reference": {
      "url": "https://...",
      "title": "document_name.pdf",
      "from": "RAG"
    }
  }
]
```

## Configuration

The script reads from `config.json` in the application root directory. Required fields:

```json
{
  "region": "us-west-2",
  "projectName": "my-project",
  "knowledge_base_id": "KB_ID",
  "sharing_url": "https://sharing-base-url",
  "aws": {
    "access_key_id": "optional",
    "secret_access_key": "optional",
    "session_token": "optional"
  }
}
```

- If AWS credentials are not provided, the script uses the default credential chain (IAM role, environment variables, etc.).

## Implementation Notes

- Uses Amazon Bedrock Agent Runtime `retrieve` API with vector search
- Handles `ResourceNotFoundException` by automatically looking up the knowledge base by project name
- URL-encodes S3 document names for proper linking
- Supports both S3 and web location sources
- Logs diagnostics to stderr for debugging

## Dependencies

The script requires:
- `boto3` - for AWS Bedrock API calls
- `botocore` - for AWS exception handling (included with boto3)

Install dependencies:
```bash
pip install boto3
```
