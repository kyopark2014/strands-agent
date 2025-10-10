
import logging
import sys
import mcp_config
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

from strands import Agent
from strands.multiagent import GraphBuilder, MultiAgentBase, MultiAgentResult
from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message
from strands.multiagent.base import NodeResult, Status

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

class QualityChecker(MultiAgentBase):
    """Custom node that evaluates content quality."""
    
    def __init__(self, approval_after: int = 2):
        super().__init__()
        self.approval_after = approval_after
        self.iteration = 0
        self.name = "checker"
        
    async def invoke_async(self, task, invocation_state=None, **kwargs):
        self.iteration += 1
        approved = self.iteration >= self.approval_after
        
        msg = f"✅ Iteration {self.iteration}: APPROVED" if approved else f"⚠️ Iteration {self.iteration}: NEEDS REVISION"
        
        agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text=msg)]),
            metrics=None,
            state={"approved": approved, "iteration": self.iteration}
        )
        
        return MultiAgentResult(
            status=Status.COMPLETED,
            results={self.name: NodeResult(result=agent_result, execution_time=10, status=Status.COMPLETED)},
            execution_count=1,
            execution_time=10
        )

async def run_graph_with_loop(question, containers):
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

        writer = Agent(
            name="writer",
            system_prompt="You are a content writer. Write or improve content based on the task. Keep responses concise."
        )

        finalizer = Agent(
            name="finalizer", 
            system_prompt="Polish the approved content into a professional format with a title."
        )

        checker = QualityChecker(approval_after=2)

        builder = GraphBuilder()
        builder.add_node(writer, "writer")
        builder.add_node(checker, "checker") 
        builder.add_node(finalizer, "finalizer")

        builder.add_edge("writer", "checker")

        def needs_revision(state):
            checker_result = state.results.get("checker")
            if not checker_result:
                return False
            multi_result = checker_result.result
            if hasattr(multi_result, 'results') and 'checker' in multi_result.results:
                agent_result = multi_result.results['checker'].result
                if hasattr(agent_result, 'state'):
                    return not agent_result.state.get("approved", False)
            return True
        
        def is_approved(state):
            checker_result = state.results.get("checker")
            if not checker_result:
                return False
            multi_result = checker_result.result
            if hasattr(multi_result, 'results') and 'checker' in multi_result.results:
                agent_result = multi_result.results['checker'].result
                if hasattr(agent_result, 'state'):
                    return agent_result.state.get("approved", False)
            return False
        
        builder.add_edge("checker", "writer", condition=needs_revision)
        builder.add_edge("checker", "finalizer", condition=is_approved)
        
        builder.set_entry_point("writer")
        builder.set_max_node_executions(10)
        builder.set_execution_timeout(60)
        builder.reset_on_revisit(True)

        graph = builder.build()

        result = await graph.invoke_async(question)

        # Show execution path
        logger.info(f"\nExecution path: {' -> '.join([node.node_id for node in result.execution_order])}")
        
        # Show loop statistics
        node_visits = {}
        for node in result.execution_order:
            node_visits[node.node_id] = node_visits.get(node.node_id, 0) + 1
        
        loops = [f"{node_id} ({count}x)" for node_id, count in node_visits.items() if count > 1]
        if loops:
            logger.info(f"Loops detected: {', '.join(loops)}")
        
        # Show final result
        if "finalizer" in result.results:
            logger.info(f"\n✨ Final Result:\n{result.results['finalizer'].result}")

        final_result = await show_result(result, containers)
        logger.info(f"final_result: {final_result}")

        if containers is not None:
            containers['notification'][index].markdown(final_result)

    return final_result


