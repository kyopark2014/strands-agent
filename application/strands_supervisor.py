# Sources from https://github.com/strands-agents/samples/blob/main/01-tutorials/02-multi-agent-systems/01-agent-as-tool/agent-as-tools.ipynb

import chat
import os
import logging
import sys
import strands_agent

from strands import Agent, tool
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

RESEARCH_ASSISTANT_PROMPT = """You are a specialized research assistant. Focus only on providing
factual, well-sourced information in response to research questions.
Always cite your sources when possible."""

PRODUCT_RECOMMENDATION_ASSISTANT_PROMPT = """You are a specialized product recommendation assistant.
Provide personalized product suggestions based on user preferences. Always cite your sources."""

TRIP_PLANNING_ASSISTANT_PROMPT = """You are a specialized travel planning assistant.
Create detailed travel itineraries based on user preferences."""

MAIN_SYSTEM_PROMPT = """
You are an assistant that routes queries to specialized agents:
- For research questions and factual information → Use the research_assistant tool
- For product recommendations and shopping advice → Use the product_recommendation_assistant tool
- For travel planning and itineraries → Use the trip_planning_assistant tool
- For simple questions not requiring specialized knowledge → Answer directly

Always select the most appropriate tool based on the user's query.
"""

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
        try:
            model = strands_agent.get_model()

            research_agent = Agent(
                model=model,
                system_prompt=RESEARCH_ASSISTANT_PROMPT,
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
    orchestrator = Agent(
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
