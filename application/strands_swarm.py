# Swarm agents
# motivated from: https://github.com/strands-agents/samples/blob/main/01-tutorials/02-multi-agent-systems/02-swarm-agent/swarm.ipynb

import chat
import os
import logging
import sys
import strands_agent
import re

from strands import Agent

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
async def run_swarm(question, containers):    
    global status_msg
    status_msg = []

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"(start"))    

    # Create specialized agents with different expertise
    # research agent
    if isKorean(question):
        system_prompt = (
            "당신은 정보 수집과 분석을 전문으로 하는 연구원입니다. "
            "당신의 역할은 해당 주제에 대한 사실적 정보와 연구 통찰력을 제공하는 것입니다. "
            "정확한 데이터를 제공하고 문제의 핵심적인 측면들을 파악하는 데 집중해야 합니다. "
            "다른 에이전트로부터 입력을 받을 때, 그들의 정보가 당신의 연구와 일치하는지 평가하세요. "
        )
    else:
        system_prompt = (
            "You are a Research Agent specializing in gathering and analyzing information. "
            "Your role in the swarm is to provide factual information and research insights on the topic. "
            "You should focus on providing accurate data and identifying key aspects of the problem. "
            "When receiving input from other agents, evaluate if their information aligns with your research. "
        )
    model = strands_agent.get_model()
    research_agent = Agent(
        model=model,
        system_prompt=system_prompt, 
        callback_handler=None)

    # Creative Agent
    if isKorean(question):
        system_prompt = (
            "당신은 혁신적인 솔루션 생성을 전문으로 하는 창의적 에이전트입니다. "
            "당신의 역할은 틀에 박힌 사고에서 벗어나 창의적인 접근법을 제안하는 것입니다. "
            "다른 에이전트들로부터 얻은 정보를 바탕으로 하되, 당신만의 독창적인 창의적 관점을 추가해야 합니다. "
            "다른 사람들이 고려하지 않았을 수도 있는 참신한 접근법에 집중하세요. "
        )
    else:
        system_prompt = (
            "You are a Creative Agent specializing in generating innovative solutions. "
            "Your role in the swarm is to think outside the box and propose creative approaches. "
            "You should build upon information from other agents while adding your unique creative perspective. "
            "Focus on novel approaches that others might not have considered. "
        )
    creative_agent = Agent(
        model=model,
        system_prompt=system_prompt, 
        callback_handler=None)

    # Critical Agent
    if isKorean(question):
        system_prompt = (
            "당신은 제안서를 분석하고 결함을 찾는 것을 전문으로 하는 비판적 에이전트입니다. "
            "당신의 역할은 다른 에이전트들이 제안한 해결책을 평가하고 잠재적인 문제점들을 식별하는 것입니다. "
            "제안된 해결책을 신중히 검토하고, 약점이나 간과된 부분을 찾아내며, 개선 방안을 제시해야 합니다. "
            "비판할 때는 건설적으로 하되, 최종 해결책이 견고하도록 보장하세요. "
        )
    else:
        system_prompt = (
            "You are a Critical Agent specializing in analyzing proposals and finding flaws. "
            "Your role in the swarm is to evaluate solutions proposed by other agents and identify potential issues. "
            "You should carefully examine proposed solutions, find weaknesses or oversights, and suggest improvements. "
            "Be constructive in your criticism while ensuring the final solution is robust. "
        )
    critical_agent = Agent(
        model=model,
        system_prompt=system_prompt, 
        callback_handler=None)

    # summarizer agent
    if isKorean(question):
        system_prompt = (
            "당신은 정보 종합을 전문으로 하는 요약 에이전트입니다. "
            "당신의 역할은 모든 에이전트로부터 통찰력을 수집하고 응집력 있는 최종 해결책을 만드는 것입니다."
            "최고의 아이디어들을 결합하고 비판점들을 다루어 포괄적인 답변을 만들어야 합니다. "
            "원래 질문을 효과적으로 다루는 명확하고 실행 가능한 요약을 작성하는 데 집중하세요. "
        )
    else:
        system_prompt = (
            "You are a Summarizer Agent specializing in synthesizing information. "
            "Your role in the swarm is to gather insights from all agents and create a cohesive final solution. "
            "You should combine the best ideas and address the criticisms to create a comprehensive response. "
            "Focus on creating a clear, actionable summary that addresses the original query effectively. "
        )
    summarizer_agent = Agent(
        model=model,
        system_prompt=system_prompt,
        callback_handler=None)

    # Dictionary to track messages between agents (mesh communication)
    research_messages = []
    creative_messages = []
    critical_messages = []
    summarizer_messages = []
    
    # Phase 1: Initial analysis by each specialized agent
    add_notification(containers, f"Phase 1: Initial analysis by each specialized agent")
    add_notification(containers, f"research agent")
    result = research_agent.stream_async(question)
    research_result = await show_streams(result, containers)
    logger.info(f"research_result: {research_result}")
    
    add_notification(containers, f"creative agent")
    result = creative_agent.stream_async(question)
    creative_result = await show_streams(result, containers)

    add_notification(containers, f"critical agent")
    result = critical_agent.stream_async(question)
    critical_result = await show_streams(result, containers)

    # Share results with all other agents (mesh communication)    
    creative_messages.append(f"From Research Agent: {research_result}")
    critical_messages.append(f"From Research Agent: {research_result}")
    summarizer_messages.append(f"From Research Agent: {research_result}")

    research_messages.append(f"From Creative Agent: {creative_result}")
    critical_messages.append(f"From Creative Agent: {creative_result}")
    summarizer_messages.append(f"From Creative Agent: {creative_result}")

    research_messages.append(f"From Critical Agent: {critical_result}")
    creative_messages.append(f"From Critical Agent: {critical_result}")
    summarizer_messages.append(f"From Critical Agent: {critical_result}")

    # Phase 2: Each agent refines based on input from others
    research_prompt = f"{question}\n\nConsider these messages from other agents:\n" + "\n\n".join(research_messages)
    logger.info(f"research_prompt: {research_prompt}")
    creative_prompt = f"{question}\n\nConsider these messages from other agents:\n" + "\n\n".join(creative_messages)
    # logger.info(f"creative_prompt: {creative_prompt}")
    critical_prompt = f"{question}\n\nConsider these messages from other agents:\n" + "\n\n".join(critical_messages)
    # logger.info(f"critical_prompt: {critical_prompt}")

    add_notification(containers, f"Phase 2: Each agent refines based on input from others")
    add_notification(containers, f"refined research agent")
    result = research_agent.stream_async(research_prompt)
    refined_research = await show_streams(result, containers)
    logger.info(f"refined_research: {refined_research}")

    add_notification(containers, f"refined creative agent")
    result = creative_agent.stream_async(creative_prompt)
    refined_creative = await show_streams(result, containers)

    add_notification(containers, f"refined critical agent")
    result = critical_agent.stream_async(critical_prompt)
    refined_critical = await show_streams(result, containers)

    # Share refined results with summarizer
    summarizer_messages.append(f"From Research Agent (Phase 2): {refined_research}")
    summarizer_messages.append(f"From Creative Agent (Phase 2): {refined_creative}")
    summarizer_messages.append(f"From Critical Agent (Phase 2): {refined_critical}")

    logger.info(f"summarized messages: {summarizer_messages}")

    # Final phase: Summarizer creates the final solution
    summarizer_prompt = f"""
Original query: {question}

Please synthesize the following inputs from all agents into a comprehensive final solution:

{"\n\n".join(summarizer_messages)}

Create a well-structured final answer that incorporates the research findings, 
creative ideas, and addresses the critical feedback.
"""

    add_notification(containers, f"summarizer agent")
    result = summarizer_agent.stream_async(summarizer_prompt)
    final_solution = await show_streams(result, containers)
    logger.info(f"final_solution: {final_solution}")

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"end)"))

    return final_solution