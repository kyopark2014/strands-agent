import utils
import info
import boto3
import traceback
import uuid
import json
import logging
import sys
import asyncio
import os

from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, current_time, use_aws, python_repl
from strands.agent.conversation_manager import SlidingWindowConversationManager

from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters


logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

os.environ["STRANDS_DEV"] = "true"  # Skip user confirmation for use_aws tool

model_name = "Claude 3.7 Sonnet"
model_type = "claude"
debug_mode = "Enable"
model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
models = info.get_model_info(model_name)
reasoning_mode = 'Disable'

def update(modelName, reasoningMode, debugMode):    
    global model_name, model_id, model_type, reasoning_mode, debug_mode
    
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

is_agent_initiate = False
tools = []  
status_container = None 

# MCP AWS documentation
documentation_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"])
))

# MCP Wikipedia
wikipedia_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="python", args=["application/mcp_server_wikipedia.py"])
))

def create_agent(history_mode):
    global tools, status_container
    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )

    model = get_model()

    try:
        tools = [calculator, current_time, use_aws, python_repl]
        
        # MCP AWS documentation
        with documentation_mcp_client as client:
            documentation_tools = client.list_tools_sync()
            logger.info(f"documentation_tools: {documentation_tools}")
            tools.extend(documentation_tools)
            
        # MCP Wikipedia
        with wikipedia_mcp_client as client:
            wikipedia_tools = client.list_tools_sync()
            logger.info(f"wikipedia_tools: {wikipedia_tools}")
            tools.extend(wikipedia_tools)

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

        tool_list = []
        for tool in tools:
            # MCP tool
            if hasattr(tool, 'tool_name'):
                tool_list.append(tool.tool_name)
            
            # strands_tools 
            if str(tool).startswith("<module 'strands_tools."):
                module_name = str(tool).split("'")[1].split('.')[-1]
                tool_list.append(module_name)

        logger.info(f"Tools: {tool_list}")

        if debug_mode == 'Enable':
            status_container.info(f"Tools: {tool_list}")

    except Exception as e:
        logger.error(f"Error initializing MCP clients: {e}")
        agent = Agent(
            model=model,
            system_prompt=system,
            tools=[calculator, current_time, use_aws]
        )

    return agent

async def process_streaming_response(question, message_placeholder):
    full_response = ""
    try:
        with documentation_mcp_client as doc_client, wikipedia_mcp_client as diagram_client:
            agent_stream = agent.stream_async(question)
            async for event in agent_stream:
                if "data" in event:
                    full_response += event["data"]
                    message_placeholder.markdown(full_response)
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        message_placeholder.markdown("Sorry, an error occurred while generating a response.")

    return full_response
            
def run_strands_agent(question, history_mode, st):
    global agent, is_agent_initiate, status_container  # status_container를 전역 변수로 사용
    status_container = st.empty()

    if not is_agent_initiate:
        agent = create_agent(history_mode)
        is_agent_initiate = True

    message_placeholder = st.empty()
    
    full_response = asyncio.run(process_streaming_response(question, message_placeholder))

    return full_response

