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

import strands_agent

async def run_workflow(question, containers):
    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"(start"))    

    model = strands_agent.get_model()
    researcher = Agent(
        model=model,
        system_prompt="research specialist. Find key information.", 
        callback_handler=None
    )
    analyst = Agent(
        model=model,
        system_prompt="You analyze research data and extract insights. Analyze these research findings.", 
        callback_handler=None
    )
    writer = Agent(
        model=model, 
        system_prompt="You create polished reports based on analysis. Create a report based on this analysis.",
        callback_handler=None
    )

    # Step 1: Research
    add_notification(containers, f"질문: {question}")
    query = f"다음의 질문을 분석하세요. <question>{question}</question>"
    research_stream = researcher.stream_async(query)
    research_result = await show_streams(research_stream, containers)    
    logger.info(f"research_results: {research_result}")

    # Step 2: Analysis
    add_notification(containers, f"분석: {research_result}")
    analysis = f"다음을 분석해서 필요한 데이터를 추가하고 이해하기 쉽게 분석하세요. <research>{research_result}</research>"
    analysis_stream = analyst.stream_async(analysis)
    analysis_result = await show_streams(analysis_stream, containers)    
    logger.info(f"analysis_results: {analysis_result}")

    # Step 3: Report writing
    add_notification(containers, f"보고서: {analysis_result}")
    report = f"다음의 내용을 참조하여 상세한 보고서를 작성하세요. <subject>{analysis_result}</subject>"
    report_stream = writer.stream_async(report)
    report_result = await show_streams(report_stream, containers)    
    logger.info(f"report_results: {report_result}")

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"end)"))

    return report_result

# supervisor agent
async def run_workflow_tool(question, containers):
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
