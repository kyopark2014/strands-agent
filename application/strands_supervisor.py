# agent-as-tool type 
# Sources from https://github.com/strands-agents/samples/blob/main/01-tutorials/02-multi-agent-systems/01-agent-as-tool/agent-as-tools.ipynb

import chat
import os
import logging
import sys
import strands_agent
import re
import json
import re
import chat
import mcp_config

from strands import Agent, tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from strands_tools import file_write

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("strands-agent")

index = 0
def add_notification(containers, message):
    global index
    containers['notification'][index].info(message)
    index += 1

def add_response(containers, message):
    global index
    containers['notification'][index].markdown(message)
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

os.environ["BYPASS_TOOL_CONSENT"] = "true"

async def show_streams(agent_stream, containers):
    tool_name = ""
    result = ""
    current_response = ""

    async for event in agent_stream:
        # logger.info(f"event: {event}")
        if "message" in event:
            message = event["message"]
            logger.info(f"message: {message}")

            for content in message["content"]:                
                if "text" in content:
                    logger.info(f"text: {content['text']}")
                    if chat.debug_mode == 'Enable':
                        add_response(containers, content['text'])

                    result = content['text']
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
                    logger.info(f"tool_name: {tool_name}")
                    logger.info(f"tool_result: {tool_result}")
                    if "content" in tool_result:
                        tool_content = tool_result['content']
                        for content in tool_content:
                            if "text" in content:
                                if chat.debug_mode == 'Enable':
                                    add_notification(containers, f"tool result: {content['text']}")

        if "data" in event:
            text_data = event["data"]
            current_response += text_data

            if chat.debug_mode == 'Enable':
                containers["notification"][index].markdown(current_response)
            continue
    
    return result

def create_mcp_client(mcp_server_name: str):
    config = mcp_config.load_config(mcp_server_name)
    mcp_servers = config["mcpServers"]
    
    mcp_client = None
    for server_name, server_config in mcp_servers.items():
        logger.info(f"server_name: {server_name}")
        logger.info(f"server_config: {server_config}")   

        env = server_config["env"] if "env" in server_config else None

        if server_name == mcp_server_name:
            mcp_client = MCPClient(lambda: stdio_client(
                StdioServerParameters(
                    command=server_config["command"], 
                    args=server_config["args"], 
                    env=env
                )
            ))
            break
    
    return mcp_client

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False

# supervisor agent
async def run_agent(question, containers):
    global status_msg
    status_msg = []

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"(start"))    

    @tool
    async def research_assistant(query: str) -> str:
        """
        Process and respond to research-related queries.

        Args:
            query: A research question requiring factual information

        Returns:
            A detailed research answer with citations
        """

        if isKorean(query):
            RESEARCH_ASSISTANT_PROMPT = (
                "당신은 연구 도우미 입니다.\n"
                "연구 질문에 대한 사실적인 정보를 제공하는 것에 집중해주세요.\n"
                "항상 출처를 표시해주세요."
            )
        else:
            RESEARCH_ASSISTANT_PROMPT = (
                "You are a specialized research assistant.\n"
                "Focus only on providing factual, well-sourced information in response to research questions.\n"
                "Always cite your sources when possible."
            )

        try:
            mcp_client = create_mcp_client("tavily-search")
            logger.info(f"mcp_client: {mcp_client}")
            with mcp_client as client:
                mcp_tools = client.list_tools_sync()
                logger.info(f"mcp_tools: {mcp_tools}")
                
                tools = []
                tools.extend(mcp_tools)

                tool_list = strands_agent.get_tool_list(tools)
                logger.info(f"tools loaded: {tool_list}")
        
                model = strands_agent.get_model()

                research_agent = Agent(
                    model=model,
                    system_prompt=RESEARCH_ASSISTANT_PROMPT,
                    tools=tools
                )

                agent_stream = research_agent.stream_async(query)
                result = await show_streams(agent_stream, containers)
                logger.info(f"result of research_assistant: {result}")

            return result
        except Exception as e:
            return f"Error in research assistant: {str(e)}"

    @tool
    async def product_recommendation_assistant(query: str) -> str:
        """
        Handle product recommendation queries by suggesting appropriate products.

        Args:
            query: A product inquiry with user preferences

        Returns:
            Personalized product recommendations with reasoning
        """

        if isKorean(query):
            PRODUCT_RECOMMENDATION_ASSISTANT_PROMPT = (
                "당신은 제품 추천 도우미 입니다.\n"
                "사용자의 취향에 맞는 제품을 추천해주세요.\n"
                "항상 출처를 표시해주세요."
            )
        else:
            PRODUCT_RECOMMENDATION_ASSISTANT_PROMPT = (
                "You are a specialized product recommendation assistant.\n"
                "Provide personalized product suggestions based on user preferences.\n"
                "Always cite your sources."
            )

        try:
            model = strands_agent.get_model()

            product_agent = Agent(
                model=model,
                system_prompt=PRODUCT_RECOMMENDATION_ASSISTANT_PROMPT,
            )

            agent_stream = product_agent.stream_async(query)
            result = await show_streams(agent_stream, containers)
            logger.info(f"result of research_assistant: {result}")

            return result
        except Exception as e:
            return f"Error in product recommendation: {str(e)}"

    @tool
    async def trip_planning_assistant(query: str) -> str:
        """
        Create travel itineraries and provide travel advice.

        Args:
            query: A travel planning request with destination and preferences

        Returns:
            A detailed travel itinerary or travel advice
        """

        if isKorean(query):
            TRIP_PLANNING_ASSISTANT_PROMPT = (
                "당신은 여행 도우미 입니다.\n"
                "여행 계획을 작성해주세요.\n"
                "항상 출처를 표시해주세요."
            )
        else:
            TRIP_PLANNING_ASSISTANT_PROMPT = (
                "You are a specialized travel planning assistant.\n"
                "Create detailed travel itineraries based on user preferences.\n"
                "Always cite your sources."
            )

        try:
            model = strands_agent.get_model()

            travel_agent = Agent(
                model=model,
                system_prompt=TRIP_PLANNING_ASSISTANT_PROMPT,
            )
            
            agent_stream = travel_agent.stream_async(query)
            result = await show_streams(agent_stream, containers)
            logger.info(f"result of research_assistant: {result}")

            return result
        except Exception as e:
            return f"Error in trip planning: {str(e)}"

    # orchestrator agent
    if isKorean(question):
        MAIN_SYSTEM_PROMPT = (
            "당신은 쿼리를 전문 에이전트로 라우팅하는 어시스턴트입니다.\n"
            "연구 질문 및 사실 정보 → research_assistant 도구 사용\n"
            "제품 추천 및 쇼핑 조언 → product_recommendation_assistant 도구 사용\n"
            "여행 계획 및 일정 → trip_planning_assistant 도구 사용\n"
            "전문 지식이 필요하지 않은 간단한 질문 → 직접 답변\n"
            "항상 사용자의 쿼리에 따라 가장 적절한 도구를 선택하세요."    
        )
    else:
        MAIN_SYSTEM_PROMPT = (
            "You are an assistant that routes queries to specialized agents:\n"
            "- For research questions and factual information → Use the research_assistant tool\n"
            "- For product recommendations and shopping advice → Use the product_recommendation_assistant tool\n"
            "- For travel planning and itineraries → Use the trip_planning_assistant tool\n"
            "- For simple questions not requiring specialized knowledge → Answer directly\n"
            "Always select the most appropriate tool based on the user's query."
        )

    orchestrator = Agent(
        model=strands_agent.get_model(),
        system_prompt=MAIN_SYSTEM_PROMPT,
        tools=[
            research_assistant,
            product_recommendation_assistant,
            trip_planning_assistant,
            file_write,
        ],
    )

    orchestrator.messages = []

    agent_stream = orchestrator.stream_async(question)
    result = await show_streams(agent_stream, containers)
    
    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"end)"))

    return result
