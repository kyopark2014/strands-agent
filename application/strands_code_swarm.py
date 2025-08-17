# Source: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/swarm/

import logging
import sys

from strands import Agent
from strands.multiagent import Swarm

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

def update_tool_notification(containers, tool_index, message):
    if containers is not None:
        containers['notification'][tool_index].info(message)

tool_info_list = dict()
tool_result_list = dict()
tool_name_list = dict()

async def show_streams(agent_stream, containers):
    """streaming event handling"""
    result = ""
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

        elif "current_tool_use" in event:
            current_tool_use = event["current_tool_use"]
            logger.info(f"current_tool_use: {current_tool_use}")
            name = current_tool_use.get("name", "")
            input = current_tool_use.get("input", "")
            toolUseId = current_tool_use.get("toolUseId", "")

            text = f"name: {name}, input: {input}"
            logger.info(f"[current_tool_use] {text}")

            if toolUseId not in tool_info_list: # new tool info
                global index
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
                    tool_name = tool_name_list[toolUseId]
                    logger.info(f"[toolResult] {toolResult}, [toolUseId] {toolUseId}")
                    add_notification(containers, f"Tool Result: {str(toolResult)}")
            
        elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
            pass

        else:
            logger.info(f"event: {event}")
    
    return result



async def process_swarm_result(result, containers):
    """Process SwarmResult and display execution details"""
    final_result = ""
    
    if result and hasattr(result, 'results'):
        # Show execution summary
        add_notification(containers, f"Swarm execution completed with status: {result.status}")
        add_notification(containers, f"Total execution time: {result.execution_time}ms")
        add_notification(containers, f"Total tokens used: {result.accumulated_usage.get('totalTokens', 0)}")
        
        # Show node history
        if result.node_history:
            node_names = [node.node_id for node in result.node_history]
            add_notification(containers, f"Execution sequence: {' → '.join(node_names)}")
        
        # Process each node's result
        for i, node in enumerate(result.node_history):
            node_id = node.node_id
            if node_id in result.results:
                node_result = result.results[node_id]
                
                add_notification(containers, f"=== {node_id} (Step {i+1}) ===")
                add_notification(containers, f"Execution time: {node_result.execution_time}ms")
                add_notification(containers, f"Status: {node_result.status}")
                
                # Extract and display the agent's response
                if hasattr(node_result, 'result') and hasattr(node_result.result, 'message'):
                    message = node_result.result.message
                    if message and hasattr(message, 'get'):
                        content = message.get("content", [])
                        if content and len(content) > 0:
                            text_content = content[0].get("text", "")
                            if text_content:
                                add_response(containers, f"{node_id} response: {text_content}")
                                # Store the last response as final result
                                if i == len(result.node_history) - 1:
                                    final_result = text_content
                    else:
                        text_content = str(message)
                        add_response(containers, f"{node_id} response: {text_content}")
                        if i == len(result.node_history) - 1:
                            final_result = text_content
                else:
                    text_content = str(node_result.result)
                    add_response(containers, f"{node_id} response: {text_content}")
                    if i == len(result.node_history) - 1:
                        final_result = text_content
    else:
        final_result = "Swarm execution completed but no result found"
        add_notification(containers, final_result)
    
    return final_result

async def run_code_swarm(question, containers):
    global status_msg
    status_msg = []

    global index
    index = 0

    # Create specialized agents for the swarm
    researcher = Agent(
        name="researcher", 
        system_prompt="You are a research specialist who analyzes requirements and provides detailed specifications."
    )
    coder = Agent(
        name="coder", 
        system_prompt="You are a coding specialist who implements solutions based on specifications."
    )
    reviewer = Agent(
        name="reviewer", 
        system_prompt="You are a code review specialist who ensures code quality and best practices."
    )
    architect = Agent(
        name="architect", system_prompt="You are a system architecture specialist who designs scalable and maintainable solutions."
    )

    # Create a swarm with these agents
    swarm = Swarm(
        [researcher, coder, reviewer, architect],
        max_handoffs=20,
        max_iterations=20,
        execution_timeout=900.0,  # 15 minutes
        node_timeout=300.0,       # 5 minutes per agent
        repetitive_handoff_detection_window=8,  # There must be >= 3 unique agents in the last 8 handoffs
        repetitive_handoff_min_unique_agents=3
    )

    

    # # Create a coder agent that uses the swarm as a tool
    # system_prompt = (
    #     "당신은 코드 개발 프로젝트를 관리하는 전문가입니다. "
    #     "연구, 설계, 구현, 검토 단계를 거쳐 완성된 솔루션을 제공합니다."
    # )

    # coder = Agent(
    #     name="coder",
    #     system_prompt=system_prompt,
    #     tools=[swarm]
    # )

    # Execute the swarm using invoke_async
    add_notification(containers, "Starting Code Swarm execution...")
    add_notification(containers, f"Question: {question}")
    
    # Use invoke_async since stream_async is not available for Swarm
    try:
        add_notification(containers, "Starting swarm execution...")
        
        # Execute swarm and get result
        result = await swarm.invoke_async(question)
        
        # Process the result and show execution details
        final_result = await process_swarm_result(result, containers)
        
    except Exception as e:
        logger.error(f"Swarm execution failed: {str(e)}")
        final_result = f"Error: {str(e)}"

    if containers is not None:
        containers['notification'][index].markdown(final_result)

    logger.info(f"Final result: {final_result}")

    return final_result
