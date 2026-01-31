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

- overrideSearchType에서 'HYBRID' 또는 'SEMANTIC'을 선택할 수 있습니다.

retrieve로 얻어진 결과에서 content, location 정보를 아래와 같이 추출합니다. 이때 location에서는 S3 또는 web에 대한 정보를 얻을 수 있습니다. CloudFront-S3를 이용하면 외부에서 파일에 접근할 수 있습니다.

```python
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
        uri = location["s3Location"]["uri"] if location["s3Location"]["uri"] is not None else ""
        
        name = uri.split("/")[-1]
        encoded_name = parse.quote(name)                
        url = f"{path}/{doc_prefix}{encoded_name}"
        
    elif "webLocation" in location:
        url = location["webLocation"]["url"] if location["webLocation"]["url"] is not None else ""
        name = "WEB"

json_docs.append({
    "contents": text,              
    "reference": {
        "url": url,                   
        "title": name,
        "from": "RAG"
    }
})
```

## Knowledge에서 관련 문서 조회하여 결과까지 얻는 경우

Boto3에서 제공하는 AgentsforBedrockRuntime에서 [retrieve_and_generate_stream](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate_stream.html)를 이용해 검색합니다.

```python
bedrock_agent_runtime_client = boto3.client(
    "bedrock-agent-runtime",
    region_name=bedrock_region
)

model_arn = f"arn:aws:bedrock:{region}:{account_id}:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

retrieve_response = bedrock_agent_runtime_client.retrieve_and_generate_stream(
    input={"text": query},
    retrieveAndGenerateConfiguration={
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": knowledge_base_id,
            "modelArn": model_arn,
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": number_of_results
                }
            }
        },
        "type": "KNOWLEDGE_BASE"
    }        
)
```

이때의 결과에서 stream을 추출하여 아래와 같이 활용합니다.

```python
msg = ""
for event in retrieve_response['stream']:
if "output" in event:
    text = event['output']['text']
    logger.info(f"text: {text}")
    msg += text

if "citation" in event:
    citation = event['citation']

    retrieved_references = citation.get('citation', {}).get('retrievedReferences', []) or citation.get('retrievedReferences', [])
    for ref in retrieved_references:
        content_text = url = name = ""

        if "content" in ref:
            content_text = ref["content"]["text"]

        if "location" in ref:
            location = ref["location"]
            if "s3Location" in location:
                uri = location["s3Location"]["uri"] if location["s3Location"]["uri"] is not None else ""                
                name = uri.split("/")[-1]
                encoded_name = parse.quote(name)                
                url = f"{path}/{doc_prefix}{encoded_name}"
            
            if "webLocation" in location:
                url = location["webLocation"]["url"] if location["webLocation"]["url"] is not None else ""
                name = "WEB"

        reference_doc = {
            "contents": content_text,              
            "reference": {
                "url": url,                   
                "title": name,
                "from": "RAG"
            }
        }
```


