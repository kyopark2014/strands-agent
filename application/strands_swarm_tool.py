# Swarm agents
# motivated from: https://github.com/strands-agents/samples/blob/main/01-tutorials/02-multi-agent-systems/02-swarm-agent/swarm.ipynb

import chat
import os
import logging
import sys
import strands_agent
import json

from strands import Agent, tool
from strands_tools import swarm

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("strands-agent")

streaming_index = None
index = 0
def add_notification(containers, message):
    global index

    if index == streaming_index:
        index += 1

    if containers is not None:
        containers['notification'][index].info(message)
    index += 1

def update_streaming_result(containers, message):
    global streaming_index
    streaming_index = index 

    if containers is not None:
        containers['notification'][streaming_index].markdown(message)

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
tool_info_list = dict()
tool_result_list = dict()
tool_name_list = dict()

async def run_swarm_tool(question, containers):
    global status_msg
    status_msg = []

    global index
    index = 0

    system_prompt = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )

    agent = Agent(
        model=strands_agent.get_model(),
        system_prompt=system_prompt,
        tools=[swarm]
    )

    result = agent.tool.swarm(
        task=question,
        swarm_size=3,
        coordination_pattern="collaborative"
    )    

    agent_stream = agent.stream_async(question)

    current = ""
    async for event in agent_stream:
        text = ""            
        if "data" in event:
            text = event["data"]
            logger.info(f"[data] {text}")
            current += text
            update_streaming_result(containers, current)

        elif "result" in event:
            final = event["result"]                
            message = final.message
            if message:
                content = message.get("content", [])
                result = content[0].get("text", "")
                logger.info(f"[result] {result}")
                final_result = result

        elif "current_tool_use" in event:
            current_tool_use = event["current_tool_use"]
            logger.info(f"current_tool_use: {current_tool_use}")
            name = current_tool_use.get("name", "")
            input = current_tool_use.get("input", "")
            toolUseId = current_tool_use.get("toolUseId", "")

            text = f"name: {name}, input: {input}"
            logger.info(f"[current_tool_use] {text}")

            if toolUseId not in tool_info_list: # new tool info
                index += 1
                current = ""
                logger.info(f"new tool info: {toolUseId} -> {index}")
                tool_info_list[toolUseId] = index
                tool_name_list[toolUseId] = name
                add_notification(containers, f"Tool: {name}, Input: {input}")
            else: # overwrite tool info if already exists
                logger.info(f"overwrite tool info: {toolUseId} -> {tool_info_list[toolUseId]}")
                containers['notification'][tool_info_list[toolUseId]].info(f"Tool: {name}, Input: {input}")

        elif "message" in event:
            message = event["message"]
            logger.info(f"[message] {message}")

            if "content" in message:
                content = message["content"]
                logger.info(f"tool content: {content}")
                if "toolResult" in content[0]:
                    toolResult = content[0]["toolResult"]
                    toolUseId = toolResult["toolUseId"]
                    toolContent = toolResult["content"]
                    toolResult = toolContent[0].get("text", "")
                    logger.info(f"[toolResult] {toolResult}, [toolUseId] {toolUseId}")
                    add_notification(containers, f"Tool Result: {str(toolResult)}")

                    if content:
                        logger.info(f"content: {content}")                
            
        elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
            pass

        else:
            logger.info(f"event: {event}")

    return result
