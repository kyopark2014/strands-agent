import json
import traceback
import boto3
import os
import re
import info

from botocore.config import Config

from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from urllib import parse
from pydantic.v1 import BaseModel, Field
from multiprocessing import Process, Pipe

bedrock_region = os.environ.get('bedrock_region')
projectName = os.environ.get('projectName')
path = os.environ.get('sharing_url')

knowledge_base_id = ""
numberOfDocs = 3
knowledge_base_name = projectName
s3_prefix = 'docs'
doc_prefix = s3_prefix+'/'

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False
    
selected_chat = 0
multi_region = 'Disable'
def get_chat(models, extended_thinking):
    global selected_chat

    profile = models[selected_chat]
    # print('profile: ', profile)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    number_of_models = len(models)

    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if model_type == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif model_type == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    if extended_thinking=='Enable':
        maxReasoningOutputTokens=64000
        print(f"extended_thinking: {extended_thinking}")
        thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

        parameters = {
            "max_tokens":maxReasoningOutputTokens,
            "temperature":1,            
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            "stop_sequences": [STOP_SEQUENCE]
        }
    else:
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [STOP_SEQUENCE]
        }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

def get_parallel_processing_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if model_type == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif model_type == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )        
    return chat
                
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_parallel_processing_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print(f"---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print(f"--GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()

def grade_documents_using_parallel_processing(models, question, documents):
    global selected_chat

    number_of_models = len(models)    
    filtered_docs = []    
    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, models, selected_chat))
        processes.append(process)
        
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    return filtered_docs

def get_retrieval_grader(chat):
    system = (
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(model_name, question, documents):
    print(f"###### grade_documents ######")
    print(f"start grading...")

    models = info.get_model_info(model_name)
    
    filtered_docs = []
    if multi_region == 'Enable':  # parallel processing        
        filtered_docs = grade_documents_using_parallel_processing(models, question, documents)

    else:
        # Score each doc    
        llm = get_chat(models, extended_thinking="Disable")
        retrieval_grader = get_retrieval_grader(llm)
        for i, doc in enumerate(documents):
            # print('doc: ', doc)
            
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            # print("score: ", score)
            
            grade = score.binary_score
            # print("grade: ", grade)
            
            if grade.lower() == "yes": # Document relevant
                print(f"---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            
            else: # Document not relevant
                print(f"---GRADE: DOCUMENT NOT RELEVANT---")
                continue
    
    return filtered_docs

contentList = []
def check_duplication(docs):
    global contentList
    length_original = len(docs)
    
    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:            
        if doc.page_content in contentList:
            print('duplicated!')
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)            
    length_updated_docs = len(updated_docs)   
    
    if length_original == length_updated_docs:
        print('no duplication')
    else:
        print('length of updated relevant_docs: ', length_updated_docs)
    
    return updated_docs

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")

def search_by_knowledge_base(keyword: str, top_k: int) -> str:
    print("###### search_by_knowledge_base ######")    
    
    global contentList, knowledge_base_id
    contentList = []
 
    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)
        
    relevant_docs = []

    if knowledge_base_id:    
        try:
            print(f'Attempting to retrieve from knowledge base: {knowledge_base_id}')
            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id=knowledge_base_id, 
                retrieval_config={"vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "HYBRID"   # SEMANTIC
                }},
                region_name=bedrock_region
            )
            
            docs = retriever.invoke(keyword)
            print('length of docs: ', len(docs))        
            # print('docs: ', docs)

            print('--> docs from knowledge base')
            for i, doc in enumerate(docs):
                print_doc(i, doc)
                
                content = f"{keyword}에 대해 조사한 결과는 아래와 같습니다.\n\n"
                if doc.page_content:
                    content = doc.page_content
                
                score = doc.metadata["score"]
                
                link = ""
                if "s3Location" in doc.metadata["location"]:
                    link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""
                    
                    # print('link:', link)    
                    pos = link.find(f"/{doc_prefix}")
                    name = link[pos+len(doc_prefix)+1:]
                    encoded_name = parse.quote(name)
                    # print('name:', name)
                    link = f"{path}/{doc_prefix}{encoded_name}"
                    
                elif "webLocation" in doc.metadata["location"]:
                    link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
                    name = "WEB"

                url = link
                # print('url:', url)
                
                relevant_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            'name': name,
                            'score': score,
                            'url': url,
                            'from': 'RAG'
                        },
                    )
                )    

        except Exception as e:
            knowledge_base_id = ""
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            print('Exception type:', type(e).__name__)
            print('Exception args:', e.args)
            
            # Check if it's a permission error
            if '403' in str(e) or 'Forbidden' in str(e):
                print('ERROR: Permission denied. Check IAM roles and policies.')
            elif 'ValidationException' in str(e):
                print('ERROR: Validation error. Check knowledge base configuration.')
            else:
                print('ERROR: Unexpected error during knowledge base retrieval.')
    else:
        print('ERROR: knowledge_base_id is not set. Cannot perform search.')
    
    return relevant_docs

def lambda_handler(event, context):
    print('event: ', event)
    
    function = event['function']
    print('function: ', function)

    knowledge_base_name = event["knowledge_base_name"]
    print('knowledge_base_name: ', knowledge_base_name)

    keyword = event.get('keyword')
    print('keyword: ', keyword)

    top_k = event.get('top_k')
    print('top_k: ', top_k)
    
    grading = event.get('grading')
    print('grading: ', grading)

    model_name = event.get('model_name')
    print('model_name: ', model_name)

    global multi_region
    multi_region = event.get('multi_region')
    print('multi_region: ', multi_region)

    global contentList
    contentList = []

    global knowledge_base_id
    # retrieve knowledge_base_id
    if not knowledge_base_id:
        try: 
            client = boto3.client(
                service_name='bedrock-agent',
                region_name=bedrock_region
            )   
            response = client.list_knowledge_bases(
                maxResults=20
            )
            print('(list_knowledge_bases) response: ', response)
            
            if "knowledgeBaseSummaries" in response:
                summaries = response["knowledgeBaseSummaries"]
                print(f'Found {len(summaries)} knowledge bases')
                for summary in summaries:
                    print(f'Knowledge Base: {summary["name"]} (ID: {summary["knowledgeBaseId"]})')
                    if summary["name"] == knowledge_base_name:                        
                        knowledge_base_id = summary["knowledgeBaseId"]
                        print('knowledge_base_id: ', knowledge_base_id)
                        break
                
                if not knowledge_base_id:
                    print(f'ERROR: Knowledge base with name "{knowledge_base_name}" not found!')
                    print('Available knowledge bases:')
                    for summary in summaries:
                        print(f'  - {summary["name"]}')
            else:
                print('ERROR: No knowledge bases found in response')
                print('Response keys:', response.keys() if response else 'No response')
                
        except Exception as e:
            knowledge_base_id = ""
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            print('Exception type:', type(e).__name__)
            print('Exception args:', e.args)
    
    if not knowledge_base_id:
        print('ERROR: Could not retrieve knowledge_base_id. Cannot proceed with search.')
        return {
            'response': json.dumps([], ensure_ascii=False),
            'error': 'Knowledge base not found or not accessible'
        }
    
    docs = []
    if function == 'search_rag':
        print('keyword: ', keyword)        
        relevant_docs = search_by_knowledge_base(keyword, top_k)  # retrieve
        if grading == "Enable":            
            filtered_docs = grade_documents(model_name, keyword, relevant_docs)  # grade documents            
            filtered_docs = check_duplication(filtered_docs)  # check duplication
            docs = filtered_docs            
        else:
            docs = relevant_docs

        json_docs = []
        for doc in docs:
            print('doc: ', doc)

            json_docs.append({
                "contents": doc.page_content,              
                "reference": {
                    "url": doc.metadata["url"],                   
                    "title": doc.metadata["name"],
                    "from": doc.metadata["from"]
                }
            })

        print('json_docs: ', json_docs)
        
    return {
        'response': json.dumps(json_docs, ensure_ascii=False)
    }
