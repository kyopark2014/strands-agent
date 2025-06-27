import chat
import os
import contextlib
import mcp_config
import logging
import sys
import json
from urllib import parse

from contextlib import contextmanager
from typing import Dict, List, Optional
from strands.models import BedrockModel
from strands_tools import calculator, current_time, use_aws
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("strands-agent")

tool_list = []
tools = []  

available_strands_tools = ["calculator", "current_time"]
available_mcp_tools = [
    "basic", "code interpreter", "aws document", "aws cost", "aws cli", 
    "use_aws", "aws cloudwatch", "aws storage", "image generation", "aws diagram",
    "knowledge base", "tavily", "perplexity", "ArXiv", "wikipedia", 
    "filesystem", "terminal", "text editor", "context7", "puppeteer", 
    "playwright", "firecrawl", "obsidian", "airbnb", 
    "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual", "사용자 설정"
]

index = 0
def add_notification(container, message):
    global index
    container['notification'][index].info(message)
    index += 1

def add_response(container, message):
    global index
    container['notification'][index].markdown(message)
    index += 1

status_msg = []
def get_status_msg(status):
    global status_msg
    status_msg.append(status)

    if status != "end)":
        status = " -> ".join(status_msg)
        return "[status]\n" + status + "..."
    else: 
        status = " -> ".join(status_msg)
        return "[status]\n" + status    

#########################################################
# Strands Agent 
#########################################################
def get_model():
    if chat.model_type == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif chat.model_type == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 

    if chat.model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k

    maxReasoningOutputTokens=64000
    thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

    if chat.reasoning_mode=='Enable':
        model = BedrockModel(
            boto_client_config=Config(
               read_timeout=900,
               connect_timeout=900,
               retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=chat.model_id,
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
            model_id=chat.model_id,
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
                #logger.info(f"client_name: {client_name}")
                if client_name in self.clients:
                    active_contexts.append(self.clients[client_name])

            # logger.info(f"active_contexts: {active_contexts}")
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
        # logger.info(f"config: {config}")

        # Skip if config is empty or doesn't have mcpServers
        if not config or "mcpServers" not in config:
            # logger.warning(f"No configuration found for tool: {tool}")
            continue

        # Get the first key from mcpServers
        server_key = next(iter(config["mcpServers"]))
        server_config = config["mcpServers"][server_key]
        
        name = tool  # Use tool name as client name
        command = server_config["command"]
        args = server_config["args"]
        env = server_config.get("env", {})  # Use empty dict if env is not present
        
        # logger.info(f"name: {name}, command: {command}, args: {args}, env: {env}")        

        mcp_manager.add_client(name, command, args, env)

init_mcp_clients()

def create_agent(history_mode, containers):
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
        
        for tool_name in chat.strands_tools:
            if tool_name in tool_map:
                tools.append(tool_map[tool_name])

        # MCP tools
        for mcp_tool in chat.mcp_tools:
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
            # logger.info(f"tool: {tool}")
            # MCP tool
            if hasattr(tool, 'tool_name'):
                # logger.info(f"MCP tool name: {tool.tool_name}")
                tool_list.append(tool.tool_name)
            
            # strands_tools 
            if str(tool).startswith("<module 'strands_tools."):
                module_name = str(tool).split("'")[1].split('.')[-1]
                # logger.info(f"Strands tool name: {module_name}")
                tool_list.append(module_name)

        logger.info(f"Tools: {tool_list}")

        if chat.debug_mode == 'Enable':
            containers['tool'].info(f"Tools: {tool_list}")

    except Exception as e:
        logger.error(f"Error initializing MCP clients: {e}")
        agent = Agent(
            model=model,
            system_prompt=system,
            tools=[calculator, current_time, use_aws]
        )

    return agent

async def run_agent(question, history_mode, containers):
    final_response = ""
    current_response = ""
    image_urls = []    

    global status_msg
    status_msg = []

    global agent
    if not chat.is_initiated or chat.is_updated:
        logger.info("create/update agent!")
        agent = create_agent(history_mode, containers)
        chat.is_initiated = True
        chat.is_updated = False
    else:
        if chat.debug_mode == 'Enable':
            containers['tool'].info(f"Tools: {tool_list}")
    
    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"(start"))

    with mcp_manager.get_active_clients(chat.mcp_tools) as _:
        agent_stream = agent.stream_async(question)
        
        tool_name = ""
        async for event in agent_stream:
            # logger.info(f"event: {event}")
            if "message" in event:
                message = event["message"]
                logger.info(f"message: {message}")

                for content in message["content"]:                
                    if "text" in content:
                        logger.info(f"text: {content["text"]}")
                        if chat.debug_mode == 'Enable':
                            add_response(containers, content["text"])

                        final_response = content["text"]
                        current_response = ""

                    if "toolUse" in content:
                        tool_use = content["toolUse"]
                        logger.info(f"tool_use: {tool_use}")
                        
                        tool_name = tool_use["name"]
                        input = tool_use["input"]
                        
                        logger.info(f"tool_nmae: {tool_name}, arg: {input}")
                        if chat.debug_mode == 'Enable':       
                            add_notification(containers, f"tool name: {tool_name}, arg: {input}")
                            containers['status'].info(get_status_msg(f"{tool_name}"))
                
                    if "toolResult" in content:
                        tool_result = content["toolResult"]
                        logger.info(f"tool_result: {tool_result}")
                        if "content" in tool_result:
                            tool_content = tool_result["content"]
                            for content in tool_content:
                                if "text" in content and chat.debug_mode == 'Enable':
                                    add_notification(containers, f"tool result: {content["text"]}")

                                    try:
                                        json_data = json.loads(content["text"])
                                        if isinstance(json_data, dict) and "path" in json_data:
                                            paths = json_data["path"]
                                            logger.info(f"paths: {paths}")
                                            for path in paths:
                                                if path.startswith("http"):
                                                    image_urls.append(path)
                                                    logger.info(f"Added image URL: {path}")
                                    except json.JSONDecodeError:
                                        pass
                                        
            if "event_loop_metrics" in event and \
                hasattr(event["event_loop_metrics"], "tool_metrics") and \
                "generate_image_with_colors" in event["event_loop_metrics"].tool_metrics:
                tool_info = event["event_loop_metrics"].tool_metrics["generate_image_with_colors"].tool
                if "input" in tool_info and "filename" in tool_info["input"]:
                    fname = tool_info["input"]["filename"]
                    if fname:
                        url = f"{path}/{chat.s3_image_prefix}/{parse.quote(fname)}.png"
                        if url not in image_urls:
                            image_urls.append(url)
                            logger.info(f"Added image URL: {url}")

            if "data" in event:
                text_data = event["data"]
                current_response += text_data

                if chat.debug_mode == 'Enable':
                    containers["notification"][index].markdown(current_response)
                continue

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"end)"))

    return final_response, image_urls
            