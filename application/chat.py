import utils
import info
import boto3
import traceback
import uuid
import json
import logging
import sys

import os
import contextlib
import mcp_config

from contextlib import contextmanager
from typing import Dict, List, Optional
from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, current_time, use_aws, python_repl
from strands.agent.conversation_manager import SlidingWindowConversationManager
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from urllib import parse

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

model_name = "Claude 3.7 Sonnet"
model_type = "claude"
debug_mode = "Enable"
model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
models = info.get_model_info(model_name)
reasoning_mode = 'Disable'

available_strands_tools = ["calculator", "current_time"]
available_mcp_tools = [
    "default", "code interpreter", "aws document", "aws cost", "aws cli", 
    "use_aws", "aws cloudwatch", "aws storage", "image generation", "aws diagram",
    "knowledge base", "tavily", "perplexity", "ArXiv", "wikipedia", 
    "filesystem", "terminal", "text editor", "context7", "puppeteer", 
    "playwright", "obsidian", "airbnb", 
    "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual",                
    "aws_cloudwatch_logs", "opensearch", "aws_knowledge_base", "사용자 설정"
]

mcp_selections = []
strands_tools = []
mcp_tools = []

is_updated = False
def update(modelName, reasoningMode, debugMode, selected_strands_tools, selected_mcp_tools):    
    global model_name, model_id, model_type, reasoning_mode, debug_mode, strands_tools, mcp_tools, is_updated
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]

    if reasoningMode != reasoning_mode:
        reasoning_mode = reasoningMode
        logger.info(f"reasoning_mode: {reasoning_mode}")

    if debugMode != debug_mode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")
    
    if selected_strands_tools != strands_tools:
        strands_tools = selected_strands_tools
        is_updated = True

    if selected_mcp_tools != mcp_tools:
        mcp_tools = selected_mcp_tools
        is_updated = True
        init_mcp_clients()

    logger.info(f"strands_tools: {strands_tools}")
    logger.info(f"mcp_tools: {mcp_tools}")
    
def initiate():
    global userId    
    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")
    
#########################################################
# Strands Agent 
#########################################################
def get_model():
    profile = models[0]
    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 

    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k

    maxReasoningOutputTokens=64000
    thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

    if reasoning_mode=='Enable':
        model = BedrockModel(
            boto_client_config=Config(
               read_timeout=900,
               connect_timeout=900,
               retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=64000,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 1,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            },
        )
    else:
        model = BedrockModel(
            boto_client_config=Config(
               read_timeout=900,
               connect_timeout=900,
               retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=maxOutputTokens,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 0.1,
            top_p = 0.9,
            additional_request_fields={
                "thinking": {
                    "type": "disabled"
                }
            }
        )
    return model

conversation_manager = SlidingWindowConversationManager(
    window_size=10,  
)

is_initiated = False
tools = []  

class MCPClientManager:
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        
    def add_client(self, name: str, command: str, args: List[str], env: dict[str, str] = {}) -> None:
        """Add a new MCP client"""
        self.clients[name] = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=command, args=args, env=env
            )
        ))
    
    def remove_client(self, name: str) -> None:
        """Remove an MCP client"""
        if name in self.clients:
            del self.clients[name]
    
    @contextmanager
    def get_active_clients(self, active_clients: List[str]):
        """Manage active clients context"""
        logger.info(f"active_clients: {active_clients}")
        active_contexts = []
        try:
            for client_name in active_clients:
                logger.info(f"client_name: {client_name}")
                if client_name in self.clients:
                    active_contexts.append(self.clients[client_name])

            logger.info(f"active_contexts: {active_contexts}")
            if active_contexts:
                with contextlib.ExitStack() as stack:
                    for client in active_contexts:
                        stack.enter_context(client)
                    yield
            else:
                yield
        except Exception as e:
            logger.error(f"Error in MCP client context: {e}")
            raise

# Initialize MCP client manager
mcp_manager = MCPClientManager()

# Set up MCP clients
def init_mcp_clients():
    logger.info(f"available_mcp_tools: {available_mcp_tools}")
    
    for tool in available_mcp_tools:
        config = mcp_config.load_config_by_name(tool)
        logger.info(f"config: {config}")

        # Skip if config is empty or doesn't have mcpServers
        if not config or "mcpServers" not in config:
            logger.warning(f"No configuration found for tool: {tool}")
            continue

        # Get the first key from mcpServers
        server_key = next(iter(config["mcpServers"]))
        server_config = config["mcpServers"][server_key]
        
        name = tool  # Use tool name as client name
        command = server_config["command"]
        args = server_config["args"]
        env = server_config.get("env", {})  # Use empty dict if env is not present
        
        logger.info(f"name: {name}, command: {command}, args: {args}, env: {env}")        

        mcp_manager.add_client(name, command, args, env)

init_mcp_clients()

config = utils.load_config()
print(f"config: {config}")

bedrock_region = config["region"] if "region" in config else "us-west-2"
projectName = config["projectName"] if "projectName" in config else "mcp-rag"
accountId = config["accountId"] if "accountId" in config else None

s3_prefix = 'docs'
s3_image_prefix = 'images'

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

def upload_to_s3(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )
        # Generate a unique file name to avoid collisions
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #unique_id = str(uuid.uuid4())[:8]
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"

        content_type = utils.get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 

        if content_type == "image/jpeg" or content_type == "image/png":
            s3_key = f"{s3_image_prefix}/{file_name}"
        else:
            s3_key = f"{s3_prefix}/{file_name}"
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        logger.info(f"upload response: {response}")

        #url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        url = path+'/'+s3_image_prefix+'/'+parse.quote(file_name)
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
        return None

tool_list = []
def create_agent(history_mode, tool_container):
    global tools
    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )

    model = get_model()

    try:
        # Convert tool names to actual tool objects
        tools = []
        tool_map = {
            "calculator": calculator,
            "current_time": current_time,
            "use_aws": use_aws
            # "python_repl": python_repl  # Temporarily disabled
        }
        
        for tool_name in strands_tools:
            if tool_name in tool_map:
                tools.append(tool_map[tool_name])

        # MCP tools
        for mcp_tool in mcp_tools:
            logger.info(f"mcp_tool: {mcp_tool}")
            with mcp_manager.get_active_clients([mcp_tool]) as _:
                logger.info(f"mcp_manager.clients: {mcp_manager.clients}")
                
                if mcp_tool in mcp_manager.clients:
                    client = mcp_manager.clients[mcp_tool]
                    mcp_tools_list = client.list_tools_sync()
                    logger.info(f"{mcp_tool}_tools: {mcp_tools_list}")
                    tools.extend(mcp_tools_list)

        logger.info(f"tools: {tools}")

        if history_mode == "Enable":
            logger.info("history_mode: Enable")
            agent = Agent(
                model=model,
                system_prompt=system,
                tools=tools,
                conversation_manager=conversation_manager
            )
        else:
            logger.info("history_mode: Disable")
            agent = Agent(
                model=model,
                system_prompt=system,
                tools=tools
                #max_parallel_tools=2
            )

        global tool_list
        tool_list = []
        for tool in tools:
            logger.info(f"tool: {tool}")
            # MCP tool
            if hasattr(tool, 'tool_name'):
                logger.info(f"MCP tool name: {tool.tool_name}")
                tool_list.append(tool.tool_name)
            
            # strands_tools 
            if str(tool).startswith("<module 'strands_tools."):
                module_name = str(tool).split("'")[1].split('.')[-1]
                logger.info(f"Strands tool name: {module_name}")
                tool_list.append(module_name)

        logger.info(f"Tools: {tool_list}")

        if debug_mode == 'Enable':
            tool_container.info(f"Tools: {tool_list}")

    except Exception as e:
        logger.error(f"Error initializing MCP clients: {e}")
        agent = Agent(
            model=model,
            system_prompt=system,
            tools=[calculator, current_time, use_aws]
        )

    return agent

async def run_agent(question, history_mode, tool_container, status_container, response_container, key_container):
    final_response = ""
    image_urls = []

    global agent, is_initiated, is_updated
    if not is_initiated or is_updated:
        logger.info("create/update agent!")
        agent = create_agent(history_mode, tool_container)
        is_initiated = True
        is_updated = False
    else:
        if debug_mode == 'Enable':
            tool_container.info(f"Tools: {tool_list}")

    with mcp_manager.get_active_clients(mcp_tools) as _:
        agent_stream = agent.stream_async(question)
        
        tool_name = ""
        async for event in agent_stream:
            # logger.info(f"event: {event}")
            if "message" in event:
                message = event["message"]
                logger.info(f"message: {message}")
                logger.info(f"content: {message["content"]}")

                # role = message["role"]
                # logger.info(f"role: {role}")

                for content in message["content"]:                
                    if "text" in content:
                        logger.info(f"text: {content["text"]}")
                        final_response += '\n\n' + content["text"]

                    if "toolUse" in content:
                        tool_use = content["toolUse"]
                        logger.info(f"tool_use: {tool_use}")
                        
                        tool_name = tool_use["name"]
                        input = tool_use["input"]
                        
                        logger.info(f"tool_nmae: {tool_name}, arg: {input}")
                        status_container.info(f"tool name: {tool_name}, arg:: {input}")
                
                    if "toolResult" in content:
                        tool_result = content["toolResult"]
                        logger.info(f"tool_result: {tool_result}")
                        if "content" in tool_result:
                            tool_content = tool_result["content"]
                            for content in tool_content:
                                if "text" in content:
                                    response_container.info(f"tool result: {content["text"]}")

            if "event_loop_metrics" in event and \
                hasattr(event["event_loop_metrics"], "tool_metrics") and \
                "generate_image_with_colors" in event["event_loop_metrics"].tool_metrics:
                tool_info = event["event_loop_metrics"].tool_metrics["generate_image_with_colors"].tool
                if "input" in tool_info and "filename" in tool_info["input"]:
                    fname = tool_info["input"]["filename"]
                    if fname:
                        url = f"{path}/{s3_image_prefix}/{parse.quote(fname)}.png"
                        if url not in image_urls:
                            image_urls.append(url)
                            logger.info(f"Added image URL: {url}")

            if "data" in event:
                text_data = event["data"]
                final_response += text_data
                key_container.markdown(final_response)
                continue

    return final_response, image_urls
            