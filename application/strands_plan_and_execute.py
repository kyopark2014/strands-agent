
import logging
import sys

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

async def run_plan_and_execute(question, containers):
    global status_msg
    status_msg = []

    global index
    index = 0

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
            "주어진 plan중 아직 실행되지 않은 첫번째 task를 실행하고 결과를 리턴합니다."
            "이전에 실행된 task가 있다면 그 다음 task를 실행합니다."
            "모든 task가 완료되었다면 'All tasks completed'라고 리턴합니다."
        )
    )
    replanner = Agent(
        name="replanner", 
        system_prompt=(
            "You are a replanner who replans the plan if the executor fails to execute the plan correctly."
            "주어진 plan에서 실행된 내용을 제외하고 새로운 plan을 생성합니다."
            "생성된 계획은 <plan> 태그로 감싸서 반환합니다."
            "executor가 'All tasks completed'라고 리턴했다면, <complete>synthesize</complete> 태그로 감싸서 반환합니다."
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
    builder.add_edge("executor", "replanner")
    builder.add_edge("replanner", "synthesizer", condition=lambda state: decide_next_step(state) == "synthesizer")
    builder.add_edge("replanner", "executor", condition=lambda state: decide_next_step(state) == "executor")
    
    # Build the graph
    graph = builder.build()

    # Execute task on newly built graph
    result = graph(question) 

    final_result = await show_result(result, containers)
    logger.info(f"final_result: {final_result}")

    if containers is not None:
        containers['notification'][index].markdown(final_result)

    # logger.info(f"Final result: {final_result}")

    return final_result


