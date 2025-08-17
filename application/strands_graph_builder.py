
import logging
import sys
import strands_agent

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

async def run_graph_builder(question, containers):
    global status_msg
    status_msg = []

    global index
    index = 0

    # Create specialized agents
    financial_advisor = Agent(
        name="financial_advisor", 
        system_prompt=(
            "You are a financial advisor focused on cost-benefit analysis, budget implications, and ROI calculations.\n" 
            "Engage with other experts to build comprehensive financial perspectives."
        )
    )
    technical_architect = Agent(
        name="technical_architect", 
        system_prompt=(
            "You are a technical architect who evaluates feasibility, implementation challenges, and technical risks.\n" 
            "Collaborate with other experts to ensure technical viability."
        )
    )
    market_researcher = Agent(
        name="market_researcher", 
        system_prompt=(
            "You are a market researcher who analyzes market conditions, user needs, and competitive landscape.\n"
            "Work with other experts to validate market opportunities."
        )
    )
    risk_analyst = Agent(
        name="risk_analyst", 
        system_prompt=(
            "You are a risk analyst who identifies potential risks, mitigation strategies, and compliance issues.\n" 
            "Collaborate with other experts to ensure comprehensive risk assessment."
        )
    )

    # Build the graph
    builder = GraphBuilder()

    # Add nodes
    builder.add_node(financial_advisor, "finance_expert")
    builder.add_node(technical_architect, "tech_expert")
    builder.add_node(market_researcher, "market_expert")
    builder.add_node(risk_analyst, "risk_analyst")

    # Add edges (dependencies)
    builder.add_edge("finance_expert", "tech_expert")
    builder.add_edge("finance_expert", "market_expert")
    builder.add_edge("tech_expert", "risk_analyst")
    builder.add_edge("market_expert", "risk_analyst")

    # Set entry points (optional - will be auto-detected if not specified)
    builder.set_entry_point("finance_expert")

    # Build the graph
    graph = builder.build()

    # Execute task on newly built graph
    result = graph(question) 

    final_result = await show_result(result, containers)
    logger.info(f"final_result: {final_result}")

    if containers is not None:
        containers['notification'][index].markdown(final_result)

    logger.info(f"Final result: {final_result}")

    return final_result


