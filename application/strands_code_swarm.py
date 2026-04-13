# Source: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/swarm/

import logging
import sys
import strands_agent

from strands import Agent
from strands.multiagent import Swarm
from strands_tools import swarm

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("strands-agent")

async def show_streams(agent_stream, containers):
    """streaming event handling"""
    queue = containers['queue']
    result = ""
    current = ""
    
    async for event in agent_stream:
        text = ""            
        if "data" in event:
            text = event["data"]
            logger.info(f"[data] {text}")
            current += text
            queue.stream(current)

        elif "result" in event:
            final = event["result"]                
            message = final.message
            if message:
                content = message.get("content", [])
                result = content[0].get("text", "")
                logger.info(f"[result] {result}")

        elif "current_tool_use" in event:
            current_tool_use = event["current_tool_use"]
            logger.info(f"current_tool_use: {current_tool_use}")
            name = current_tool_use.get("name", "")
            input = current_tool_use.get("input", "")
            toolUseId = current_tool_use.get("toolUseId", "")

            text = f"name: {name}, input: {input}"
            logger.info(f"[current_tool_use] {text}")

            queue.register_tool(toolUseId, name)
            queue.tool_update(toolUseId, f"Tool: {name}, Input: {input}")
            current = ""

        elif "message" in event:
            message = event["message"]
            logger.info(f"[message] {message}")

            if "content" in message:
                msg_content = message["content"]
                logger.info(f"tool content: {msg_content}")
                for item in msg_content:
                    if "toolResult" not in item:
                        continue
                    toolResult = item["toolResult"]
                    toolUseId = toolResult["toolUseId"]
                    toolContent = toolResult["content"]
                    toolResultText = toolContent[0].get("text", "")
                    tool_name = queue.get_tool_name(toolUseId)
                    logger.info(f"[toolResult] {toolResultText}, [toolUseId] {toolUseId}")
                    queue.notify(f"Tool Result: {str(toolResultText)}")
            
        elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
            pass

        else:
            logger.info(f"event: {event}")
    
    return result

async def run_code_swarm(question, containers):
    queue = containers['queue']
    queue.reset()

    # swarm tool
    agent = Agent(
        model=strands_agent.get_model(),
        system_prompt="당신은 코드 개발 프로젝트를 관리하는 전문가입니다. 연구, 설계, 구현, 검토 단계를 거쳐 완성된 솔루션을 제공합니다.",
        tools=[swarm]
    )

    result = agent.tool.swarm(
        task=question,
        agents=[
            {
                "name": "researcher",
                "system_prompt": "당신은 요구사항을 분석하고 상세한 명세서를 제공하는 연구 전문가입니다."
            },
            {
                "name": "coder",
                "system_prompt": "당신은 코드 개발 전문가입니다. 명세서를 바탕으로 코드를 작성합니다."
            },
            {
                "name": "reviewer",
                "system_prompt": "당신은 코드 검토 전문가입니다. 코드 품질과 최적화를 보장합니다."
            },
            {
                "name": "architect",
                "system_prompt": "당신은 시스템 아키텍처 전문가입니다. 확장성과 유지보수성을 고려한 솔루션을 설계합니다."
            }
        ],
        coordination_pattern="collaborative"
    )    

    agent_stream = agent.stream_async(question)

    current = ""
    final_result = ""
    
    async for event in agent_stream:
        text = ""            
        if "data" in event:
            text = event["data"]
            logger.info(f"[data] {text}")
            current += text
            queue.stream(current)

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

            queue.register_tool(toolUseId, name)
            queue.tool_update(toolUseId, f"Tool: {name}, Input: {input}")
            current = ""

        elif "message" in event:
            message = event["message"]
            logger.info(f"[message] {message}")

            if "content" in message:
                msg_content = message["content"]
                logger.info(f"tool content: {msg_content}")
                for item in msg_content:
                    if "toolResult" not in item:
                        continue
                    toolResult = item["toolResult"]
                    toolUseId = toolResult["toolUseId"]
                    toolContent = toolResult["content"]
                    toolResultText = toolContent[0].get("text", "")
                    tool_name = queue.get_tool_name(toolUseId)
                    logger.info(f"[toolResult] {toolResultText}, [toolUseId] {toolUseId}")
                    queue.notify(f"Tool Result: {str(toolResultText)}")
            
        elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
            pass

        else:
            logger.info(f"event: {event}")

    if containers is not None:
        queue.result(final_result)

    logger.info(f"Final result: {final_result}")

    return final_result
