# Workflow agents
# motivated from: https://strandsagents.com/latest/user-guide/concepts/multi-agent/workflow/terms/workflow-agent/

import chat
import os
import logging
import sys
import strands_agent
import json

from strands import Agent, tool
from strands_tools import workflow

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

# supervisor agent
async def run_agent(question, containers):
    global status_msg
    status_msg = []

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"(start"))    

    system_prompt = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )

    agent = Agent(
        model=strands_agent.get_model(),
        system_prompt=system_prompt,
        tools=[workflow]
    )

    agent.tool.workflow(
        action="create",
        workflow_id="data_analysis",
        tasks=[
            {
                "task_id": "data_extraction",
                "description": "Extract key financial data from the quarterly report",
                "system_prompt": "You extract and structure financial data from reports.",
                "priority": 5
            },
            {
                "task_id": "trend_analysis",
                "description": "Analyze trends in the data compared to previous quarters",
                "dependencies": ["data_extraction"],
                "system_prompt": "You identify trends in financial time series.",
                "priority": 3
            },
            {
                "task_id": "report_generation",
                "description": "Generate a comprehensive analysis report",
                "dependencies": ["trend_analysis"],
                "system_prompt": "You create clear financial analysis reports.",
                "priority": 2
            }
        ]
    ) 

    agent.tool.workflow(action="start", workflow_id="data_analysis")

    status = agent.tool.workflow(action="status", workflow_id="data_analysis")
    logger.info(f"status of workflow: {status}")

    #agent_stream = agent.stream_async(question)
    #result = await show_streams(agent_stream, containers)
    
    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"end)"))

    return status
