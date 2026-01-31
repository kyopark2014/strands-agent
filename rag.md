# RAG의 구현

## Knowledge에서 관련된 문서를 조회

Boto3에서 제공하는 AgentsforBedrockRuntime에서 [retrieve](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve.html)를 이용해 검색합니다.


```python
bedrock_agent_runtime_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=bedrock_region
    )
    
response = bedrock_agent_runtime_client.retrieve(
    retrievalQuery={"text": query},
    knowledgeBaseId=knowledge_base_id,
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": number_of_results},
        },
    )
```
