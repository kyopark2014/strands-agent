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
async def run_swarm_tool(question, containers):
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
        tools=[swarm]
    )

    result = agent.tool.swarm(
        task=question,
        swarm_size=2,
        coordination_pattern="collaborative"
    )    
    logger.info(f"result of swarm: {result}")

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"end)"))

    texts = []
    for i, content in enumerate(result["content"]):
        logger.info(f"content[{i}]: {content}")
        if "text" in content:
            texts.append(content["text"])

    swarm_result = texts[-1]
    logger.info(f"swarm_result: {swarm_result}")
    
    messages = []
    if "🌟 Collective Knowledge:" in swarm_result:
        json_results = swarm_result.split("🌟 Collective Knowledge:")[1].strip()
        logger.info(f"JSON results: {json_results}")
        
        try:
            json_data = json.loads(json_results)
            for json_result in json_data:
                content_text = json_result["content"]
                if "Metrics:" in content_text:
                    content_text = content_text.split("Metrics:")[0].strip()
                
                content = json_result["agent_id"]+': '+content_text
                logger.info(f"content: {content}")
                add_notification(containers, content)

                messages.append(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            add_notification(containers, json_results)
    else:
        json_results = swarm_result
        logger.info("🌟 Collective Knowledge: 패턴을 찾을 수 없습니다.")

    # summarizer agents
    if chat.isKorean(question):
        summarizer_prompt = f"""
질문: <question>{question}</question>

아래 에이전트들의 생각을 종합하여 최종 답변을 생성하세요. 
<opinion>{"\n\n".join(messages)}</opinion>
"""
    else:
        summarizer_prompt = f"""
Original query: {question}

Please synthesize the following inputs from all agents into a comprehensive final solution:

{"\n\n".join(messages)}

Create a well-structured final answer that incorporates the research findings, 
creative ideas, and addresses the critical feedback.
"""
    
    model = strands_agent.get_model()
    summarizer_agent = Agent(
        model=model,
        system_prompt=summarizer_prompt,
    )    
    agent_stream = summarizer_agent.stream_async(question)
    result = await show_streams(agent_stream, containers)
    logger.info(f"summarized result from swarm agents: {result}")

    return result
