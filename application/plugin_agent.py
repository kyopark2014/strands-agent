import os
import chat
import logging
import sys
import chat
import plugin
import strands_agent
from typing import Literal, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger("plugin-agent")


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")
PLUGINS_DIR = os.path.join(WORKING_DIR, "plugins")


async def run_plugin_agent(query: str, strands_tools: list[str], mcp_servers: list[str], plugin_name: Optional[str], containers: dict):
    """Run the plugin agent with streaming and tool notifications. Uses chat module for UI callbacks."""    

    global tool_list
    tool_list = []
    chat.index = 0
    chat.tool_info_list.clear()
    chat.tool_name_list.clear()

    image_url = []
    references = []
    index = 0

    command = None
    if plugin.is_command(query, plugin_name):
        command = query.split(" ")[0].lstrip("/")
        logger.info(f"command: {command}")

    # initiate agent
    await strands_agent.initiate_agent(
        system_prompt=None,
        strands_tools=strands_tools,
        mcp_servers=mcp_servers,
        plugin_name=plugin_name,
        command=command
    )
    logger.info(f"tool_list: {tool_list}")

    # run agent
    final_result = current = ""
    with strands_agent.mcp_manager.get_active_clients(mcp_servers) as _:
        agent_stream = strands_agent.agent.stream_async(query)

        async for event in agent_stream:
            text = ""
            if "data" in event:
                text = event["data"]
                logger.info(f"[data] {text}")
                current += text
                chat.index = index
                chat.update_streaming_result(containers, current)

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
                input_val = current_tool_use.get("input", "")
                toolUseId = current_tool_use.get("toolUseId", "")

                text = f"name: {name}, input: {input_val}"
                logger.info(f"[current_tool_use] {text}")

                if toolUseId not in chat.tool_info_list:
                    index += 1
                    current = ""
                    logger.info(f"new tool info: {toolUseId} -> {index}")
                    chat.tool_info_list[toolUseId] = index
                    chat.tool_name_list[toolUseId] = name
                    chat.index = index
                    chat.add_notification(containers, f"Tool: {name}, Input: {input_val}")
                    index = chat.index
                else:
                    logger.info(f"overwrite tool info: {toolUseId} -> {chat.tool_info_list[toolUseId]}")
                    containers['notification'][chat.tool_info_list[toolUseId]].info(f"Tool: {name}, Input: {input_val}")

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
                        tool_name = chat.tool_name_list[toolUseId]
                        logger.info(f"[toolResult] {toolResult}, [toolUseId] {toolUseId}")
                        chat.add_notification(containers, f"Tool Result: {str(toolResult)}")

                        content, urls, refs = chat.get_tool_info(tool_name, toolResult)
                        if refs:
                            for r in refs:
                                references.append(r)
                            logger.info(f"refs: {refs}")
                        if urls:
                            for url in urls:
                                image_url.append(url)
                            logger.info(f"urls: {urls}")

                        if content:
                            logger.info(f"content: {content}")

            elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
                pass

            else:
                logger.info(f"event: {event}")

        if references:
            ref = "\n\n### Reference\n"
            for i, reference in enumerate(references):
                content = reference['content'][:100].replace("\n", "")
                ref += f"{i+1}. [{reference['title']}]({reference['url']}), {content}...\n"
            final_result += ref

        if containers is not None:
            containers['notification'][chat.index].markdown(final_result)

    return final_result, image_url

