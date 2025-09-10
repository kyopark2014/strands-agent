
import logging
import sys
import mcp_config
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

from strands.multiagent import GraphBuilder
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

streaming_index = None

def update_streaming_result(containers, message):
    global streaming_index
    streaming_index = index 

    if containers is not None:
        containers['notification'][streaming_index].markdown(message)

def update_tool_notification(containers, tool_index, message):
    if containers is not None:
        containers['notification'][tool_index].info(message)

tool_info_list = dict()
tool_result_list = dict()
tool_name_list = dict()

async def show_result(graph_result, containers):
    """Batch processing for GraphResult object"""
    result = ""
    
    # Debug: Log the GraphResult object structure
    logger.info(f"GraphResult type: {type(graph_result)}")
    logger.info(f"GraphResult attributes: {[attr for attr in dir(graph_result) if not attr.startswith('_')]}")
    
    # Process execution order information
    if hasattr(graph_result, 'execution_order'):
        add_notification(containers, "=== Execution Order ===")
        for node in graph_result.execution_order:
            add_notification(containers, f"Executed: {node.node_id}")
    
    # Process performance metrics
    if hasattr(graph_result, 'total_nodes'):
        add_notification(containers, f"Total nodes: {graph_result.total_nodes}")
    if hasattr(graph_result, 'completed_nodes'):
        add_notification(containers, f"Completed nodes: {graph_result.completed_nodes}")
    if hasattr(graph_result, 'failed_nodes'):
        add_notification(containers, f"Failed nodes: {graph_result.failed_nodes}")
    if hasattr(graph_result, 'execution_time'):
        add_notification(containers, f"Execution time: {graph_result.execution_time}ms")
    if hasattr(graph_result, 'accumulated_usage'):
        add_notification(containers, f"Token usage: {graph_result.accumulated_usage}")
    
    # Process specific node results and combine them
    if hasattr(graph_result, 'results'):
        add_notification(containers, "=== Individual Node Results ===")
        node_results = []
        for node_id, node_result in graph_result.results.items():
            if hasattr(node_result, 'result'):
                node_content = f"{node_id}: {node_result.result}"
                add_notification(containers, node_content)
                node_results.append(node_content)
        
        # Combine individual node results as the final result
        if node_results:
            result = "\n\n".join(node_results)
            logger.info(f"Combined result from individual nodes: {result}")
            update_streaming_result(containers, result)
    
    return result

def get_tool_list(tools):
    tool_list = []
    for tool in tools:
        if hasattr(tool, 'tool_name'):  # MCP tool
            tool_list.append(tool.tool_name)
        elif hasattr(tool, 'name'):  # MCP tool with name attribute
            tool_list.append(tool.name)
        elif hasattr(tool, '__name__'):  # Function or module
            tool_list.append(tool.__name__)
        elif str(tool).startswith("<module 'strands_tools."):   
            module_name = str(tool).split("'")[1].split('.')[-1]
            tool_list.append(module_name)
        else:
            # For MCP tools that might have different structure
            tool_str = str(tool)
            if 'MCPAgentTool' in tool_str:
                # Try to extract tool name from MCP tool
                try:
                    if hasattr(tool, 'tool'):
                        tool_list.append(tool.tool.name)
                    else:
                        tool_list.append(f"MCP_Tool_{len(tool_list)}")
                except:
                    tool_list.append(f"MCP_Tool_{len(tool_list)}")
            else:
                tool_list.append(str(tool))
    return tool_list

debug_mode = 'Enable'
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
                    if debug_mode == 'Enable':
                        add_response(containers, content['text'])

                    result = content['text']
                    current_response = ""

                if "toolUse" in content:
                    tool_use = content["toolUse"]
                    logger.info(f"tool_use: {tool_use}")
                    
                    tool_name = tool_use["name"]
                    input = tool_use["input"]
                    
                    logger.info(f"tool_nmae: {tool_name}, arg: {input}")
                    if debug_mode == 'Enable':       
                        add_notification(containers, f"tool name: {tool_name}, arg: {input}")
                        containers['status'].info(get_status_msg(f"{tool_name}"))
            
                if "toolResult" in content:
                    tool_result = content["toolResult"]
                    logger.info(f"tool_name: {tool_name}")
                    logger.info(f"tool_result: {tool_result}")
                    logger.info(f"tool_result status: {tool_result.get('status', 'unknown')}")
                    logger.info(f"tool_result toolUseId: {tool_result.get('toolUseId', 'unknown')}")
                    
                    if "content" in tool_result:
                        tool_content = tool_result['content']
                        logger.info(f"tool_content length: {len(tool_content)}")
                        for i, content in enumerate(tool_content):
                            logger.info(f"content[{i}]: {content}")
                            if "text" in content:
                                text_value = content['text']
                                logger.info(f"text_value: {text_value}")
                                logger.info(f"text_value type: {type(text_value)}")
                                
                                # 코루틴 객체 문자열인지 확인
                                if isinstance(text_value, str) and '<coroutine object' in text_value:
                                    logger.info("Detected coroutine string, tool may still be executing...")
                                    if debug_mode == 'Enable':
                                        add_notification(containers, f"tool result: Tool execution in progress...")
                                else:
                                    if debug_mode == 'Enable':
                                        add_notification(containers, f"tool result: {text_value}")

        if "data" in event:
            text_data = event["data"]
            current_response += text_data

            if debug_mode == 'Enable':
                containers["notification"][index].markdown(current_response)
            continue
    
    logger.info(f"show_streams completed, final result: {result}")
    logger.info(f"show_streams result type: {type(result)}")
    return result

async def run_plan_and_execute(question, containers):
    global status_msg
    status_msg = []

    global index
    index = 0

    tool = "tavily-search"
    config = mcp_config.load_config(tool)
    mcp_servers = config["mcpServers"]
    logger.info(f"mcp_servers: {mcp_servers}")

    mcp_client = None
    for server_name, server_config in mcp_servers.items():
        logger.info(f"server_name: {server_name}")
        logger.info(f"server_config: {server_config}")
        env = server_config["env"] if "env" in server_config else None

        mcp_client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=server_config["command"], 
                args=server_config["args"], 
                env=env
            )
        ))
        break

    with mcp_client as client:
        mcp_tools = client.list_tools_sync()
        logger.info(f"mcp_tools: {mcp_tools}")
        
        tools = []
        tools.extend(mcp_tools)

        tool_list = get_tool_list(tools)
        logger.info(f"tools loaded: {tool_list}")

        # Create specialized agents
        planner = Agent(
            name="plan", 
            system_prompt=(
                "For the given objective, come up with a simple step by step plan."
                "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps."
                "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."
                "생성된 계획은 <plan> 태그로 감싸서 반환합니다."
            )
        )

        #agent_stream = await planner.stream_async(question)
        #result = await show_result(agent_stream, containers)
        plan = planner(question)
        logger.info(f"planner result: {plan}")

        executor = Agent(
            name="executor", 
            system_prompt=(
                "You are an executor who executes the plan."
                "주어진 질문에 답변하기 위하여 다음의 plan을 순차적으로 실행합니다."
                "tavily-search 도구를 사용하여 정보를 수집합니다."
                f"<plan>{plan}</plan>"
            ),
            tools=tools
        )

        result = executor(question)
        
        if containers is not None:
            containers['notification'][index].markdown(result)

    return result

# currently cyclic graph is not supported (Sep. 10 2025)
async def run_plan_and_execute_with_graph(question, containers):
    global status_msg
    status_msg = []

    global index
    index = 0

    tool = "tavily-search"
    config = mcp_config.load_config(tool)
    mcp_servers = config["mcpServers"]
    logger.info(f"mcp_servers: {mcp_servers}")

    mcp_client = None
    for server_name, server_config in mcp_servers.items():
        logger.info(f"server_name: {server_name}")
        logger.info(f"server_config: {server_config}")
        env = server_config["env"] if "env" in server_config else None

        mcp_client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=server_config["command"], 
                args=server_config["args"], 
                env=env
            )
        ))
        break

    with mcp_client as client:
        mcp_tools = client.list_tools_sync()
        logger.info(f"mcp_tools: {mcp_tools}")
        
        tools = []
        tools.extend(mcp_tools)

        tool_list = get_tool_list(tools)
        logger.info(f"tools loaded: {tool_list}")

        # Create specialized agents
        planner = Agent(
            name="plan", 
            system_prompt=(
                "For the given objective, come up with a simple step by step plan."
                "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps."
                "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."
                "생성된 계획은 <plan> 태그로 감싸서 반환합니다."
            )
        )

        executor = Agent(
            name="executor", 
            system_prompt=(
                "You are an executor who executes the plan."
                "주어진 plan을 순차적으로 실행합니다."
                # "모든 task가 완료되었다면 'All tasks completed'라고 리턴합니다."
            ),
            tools=tools
        )
        
        replanner = Agent(
            name="replanner", 
            system_prompt=(
                "You are a replanner who replans the plan if the executor fails to execute the plan correctly."
                "주어진 plan에서 실행된 내용을 제외하고 새로운 plan을 생성합니다."
                "생성된 계획은 <plan> 태그로 감싸서 반환합니다."
            )
        )
        synthesizer = Agent(
            name="synthesizer", 
            system_prompt=(
                "You are a synthesizer who synthesizes the final result."
                "You should synthesize the final result based on the plan and the executor's result."
                "You should return the synthesized final result."
            )
        )

        def decide_next_step(state):
            print(f"===== decide_next_step CALLED =====")
            print(f"state: {state}")
            
            # 실행 횟수 확인
            if hasattr(state, 'results'):
                print(f"[DEBUG] Current results keys: {list(state.results.keys())}")
                for key, result in state.results.items():
                    print(f"[DEBUG] {key}: {result}")
            
            replanner_result = state.results.get("replanner")
            print(f"replanner_result: {replanner_result}")

            if not replanner_result:
                print("[DEBUG] No replanner result, going to executor")
                return "executor"

            result_text = str(replanner_result.result)
            print(f"result_text: {result_text}")

            if "<complete>" in result_text:
                should_synthesize = "synthesize" in result_text.lower()
                print(f"[DEBUG] Found <complete>, should synthesize: {should_synthesize}")
                if should_synthesize:
                    print("[DEBUG] Going to synthesizer")
                    return "synthesizer"
                else:
                    print("[DEBUG] Going to executor")
                    return "executor"
            else:
                print("[DEBUG] No <complete> found, going to executor")
                return "executor"

        # Build the graph
        builder = GraphBuilder()

        # Add nodes
        builder.add_node(planner, "planner")
        builder.add_node(executor, "executor")
        builder.add_node(replanner, "replanner")
        builder.add_node(synthesizer, "synthesizer")

        # Set entry points (optional - will be auto-detected if not specified)
        builder.set_entry_point("planner")

        # Add edges (dependencies)
        builder.add_edge("planner", "executor")

        # builder.add_edge("executor", "replanner")
        # builder.add_edge("replanner", "synthesizer", condition=lambda state: decide_next_step(state) == "synthesizer")
        # builder.add_edge("replanner", "executor", condition=lambda state: decide_next_step(state) == "executor")
        
        # Build the graph
        graph = builder.build()

        result = await graph.invoke_async(question)
        final_result = await show_result(result, containers)
        logger.info(f"final_result: {final_result}")

        if containers is not None:
            containers['notification'][index].markdown(final_result)

    return final_result


