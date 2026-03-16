#!/usr/bin/env python3
"""
Knowledge base retrieve script using Amazon Bedrock RAG
"""

import boto3
import logging
import sys
import os
import json
from urllib import parse
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("retrieve")

script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
config_path = os.path.join(app_dir, "config.json")

NUMBER_OF_RESULTS = 5
DOC_PREFIX = "docs/"


def load_config():
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_bedrock_client(config):
    region = config.get('region', 'us-west-2')

    return boto3.client(
        "bedrock-agent-runtime",
        region_name=region
    )

def update_knowledge_base_id(config):
    """Look up knowledge base ID by project name when the current ID is stale."""
    region = config.get('region', 'us-west-2')
    project_name = config.get('projectName')

    agent_client = boto3.client("bedrock-agent", region_name=region)
    kb_list = agent_client.list_knowledge_bases()

    for kb in kb_list.get("knowledgeBaseSummaries", []):
        if kb["name"] == project_name:
            new_id = kb["knowledgeBaseId"]
            config['knowledge_base_id'] = new_id
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logger.info(f"Updated knowledge_base_id to: {new_id}")
            return new_id

    logger.error(f"Could not find knowledge base with name: {project_name}")
    return None


def retrieve(query):
    config = load_config()
    knowledge_base_id = config.get('knowledge_base_id')
    sharing_url = config.get('sharing_url', '')
    client = create_bedrock_client(config)

    retrieval_params = {
        "retrievalQuery": {"text": query},
        "knowledgeBaseId": knowledge_base_id,
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {"numberOfResults": NUMBER_OF_RESULTS},
        },
    }

    try:
        response = client.retrieve(**retrieval_params)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            logger.warning(f"ResourceNotFoundException: {e}")
            new_id = update_knowledge_base_id(config)
            if new_id:
                retrieval_params["knowledgeBaseId"] = new_id
                response = client.retrieve(**retrieval_params)
                logger.info("Retry successful after updating knowledge_base_id")
            else:
                raise
        else:
            raise

    retrieval_results = response.get("retrievalResults", [])

    json_docs = []
    for result in retrieval_results:
        text = url = name = None

        if "content" in result:
            content = result["content"]
            if "text" in content:
                text = content["text"]

        if "location" in result:
            location = result["location"]
            if "s3Location" in location:
                uri = location["s3Location"].get("uri", "")
                name = uri.split("/")[-1]
                encoded_name = parse.quote(name)
                url = f"{sharing_url}/{DOC_PREFIX}{encoded_name}"
            elif "webLocation" in location:
                url = location["webLocation"].get("url", "")
                name = "WEB"

        json_docs.append({
            "contents": text,
            "reference": {
                "url": url,
                "title": name,
                "from": "RAG"
            }
        })

    logger.info(f"Retrieved {len(json_docs)} results")
    return json.dumps(json_docs, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python retrieve_search.py <keyword>")
        sys.exit(1)

    keyword = sys.argv[1]
    result = retrieve(keyword)
    print(result)
