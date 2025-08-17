# Source: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/swarm/

import logging
import sys
import asyncio
import re
from typing import AsyncIterator, Any

from strands import Agent
from strands.multiagent import Swarm, SwarmResult
from strands.types.content import ContentBlock
from strands.types.event_loop import Usage, Metrics
from opentelemetry import trace as trace_api
import time

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
        # Use different styling based on message content
        if "🔧" in message:  # Tool usage
            containers['notification'][index].success(message)
        elif "🔄" in message:  # Handoff
            containers['notification'][index].warning(message)
        elif "📤" in message:  # Tool result
            containers['notification'][index].info(message)
        elif "💬" in message:  # Message
            containers['notification'][index].info(message)
        elif "❌" in message:  # Error
            containers['notification'][index].error(message)
        else:
            containers['notification'][index].info(message)
    index += 1

def update_streaming_result(containers, message):
    global streaming_index
    streaming_index = index 

    if containers is not None:
        containers['notification'][streaming_index].markdown(message)

def add_response(containers, message):
    global index
    # Use markdown with better formatting for responses
    formatted_message = f"**{message}**"
    containers['notification'][index].markdown(formatted_message)
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


class StreamingSwarm(Swarm):
    """Swarm with streaming support for real-time updates"""
    
    async def stream_async(self, task: str | list[ContentBlock], **kwargs: Any) -> AsyncIterator[Any]:
        """Stream swarm execution events in real-time"""
        logger.debug("starting streaming swarm execution")

        # Initialize swarm state with configuration
        initial_node = next(iter(self.nodes.values()))  # First SwarmNode
        # Use the parent class's state initialization
        self.state = self._create_swarm_state(initial_node, task)
        
        start_time = time.time()
        span = self.tracer.start_multiagent_span(task, "swarm")
        
        with trace_api.use_span(span, end_on_exit=True):
            try:
                logger.debug("current_node=<%s> | starting streaming swarm execution with node", self.state.current_node.node_id)
                
                # Yield initial event
                yield {
                    "type": "swarm_start",
                    "current_node": self.state.current_node.node_id,
                    "task": task
                }
                
                # Execute swarm with streaming
                async for event in self._stream_execute_swarm():
                    yield event
                    
            except Exception as e:
                logger.exception("streaming swarm execution failed")
                self.state.completion_status = "FAILED"
                yield {
                    "type": "error",
                    "error": str(e)
                }
                raise
            finally:
                self.state.execution_time = round((time.time() - start_time) * 1000)
                
                # Yield final result
                final_result = self._build_result()
                yield {
                    "type": "swarm_complete",
                    "result": final_result
                }

    def _create_swarm_state(self, initial_node, task):
        """Create swarm state with proper initialization"""
        # Create a simple state object with required attributes
        class SimpleSwarmState:
            def __init__(self, current_node, task, shared_context):
                self.current_node = current_node
                self.task = task
                self.completion_status = "EXECUTING"
                self.shared_context = shared_context
                self.node_history = []
                self.start_time = time.time()
                self.results = {}
                self.accumulated_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
                self.accumulated_metrics = {"latencyMs": 0}
                self.execution_time = 0
                self.handoff_message = None
                
            def should_continue(self, **kwargs):
                # Simple continuation logic
                if len(self.node_history) >= kwargs.get('max_handoffs', 20):
                    return False, f"Max handoffs reached: {kwargs.get('max_handoffs', 20)}"
                if len(self.node_history) >= kwargs.get('max_iterations', 20):
                    return False, f"Max iterations reached: {kwargs.get('max_iterations', 20)}"
                elapsed = time.time() - self.start_time
                if elapsed > kwargs.get('execution_timeout', 900.0):
                    return False, f"Execution timed out: {kwargs.get('execution_timeout', 900.0)}s"
                return True, "Continuing"
                
            def add_context(self, node, key: str, value: Any) -> None:
                """Add context to shared context."""
                if not hasattr(self.shared_context, 'context'):
                    self.shared_context.context = {}
                if node.node_id not in self.shared_context.context:
                    self.shared_context.context[node.node_id] = {}
                self.shared_context.context[node.node_id][key] = value
        
        return SimpleSwarmState(initial_node, task, self.shared_context)

    async def _stream_execute_swarm(self) -> AsyncIterator[Any]:
        """Stream swarm execution with real-time updates"""
        try:
            # Main execution loop
            while True:
                if self.state.completion_status != "EXECUTING":
                    reason = f"Completion status is: {self.state.completion_status}"
                    logger.debug("reason=<%s> | stopping execution", reason)
                    break

                should_continue, reason = self.state.should_continue(
                    max_handoffs=self.max_handoffs,
                    max_iterations=self.max_iterations,
                    execution_timeout=self.execution_timeout,
                    repetitive_handoff_detection_window=self.repetitive_handoff_detection_window,
                    repetitive_handoff_min_unique_agents=self.repetitive_handoff_min_unique_agents,
                )
                if not should_continue:
                    self.state.completion_status = "FAILED"
                    logger.debug("reason=<%s> | stopping execution", reason)
                    yield {
                        "type": "swarm_stopped",
                        "reason": reason
                    }
                    break

                # Get current node
                current_node = self.state.current_node
                if not current_node or current_node.node_id not in self.nodes:
                    logger.error("node=<%s> | node not found", current_node.node_id if current_node else "None")
                    self.state.completion_status = "FAILED"
                    yield {
                        "type": "error",
                        "error": f"Node {current_node.node_id if current_node else 'None'} not found"
                    }
                    break

                logger.debug(
                    "current_node=<%s>, iteration=<%d> | executing node",
                    current_node.node_id,
                    len(self.state.node_history) + 1,
                )

                # Yield node start event
                yield {
                    "type": "node_start",
                    "node_id": current_node.node_id,
                    "iteration": len(self.state.node_history) + 1
                }

                # Execute node with streaming
                try:
                    async for event in self._stream_execute_node(current_node, self.state.task):
                        yield event

                    self.state.node_history.append(current_node)

                    logger.debug("node=<%s> | node execution completed", current_node.node_id)

                    # Check if the current node is still the same after execution
                    # If it is, then no handoff occurred and we consider the swarm complete
                    if self.state.current_node == current_node:
                        logger.debug("node=<%s> | no handoff occurred, marking swarm as complete", current_node.node_id)
                        self.state.completion_status = "COMPLETED"
                        yield {
                            "type": "node_complete",
                            "node_id": current_node.node_id,
                            "final": True
                        }
                        break
                    else:
                        # Handoff occurred
                        yield {
                            "type": "handoff",
                            "from_node": current_node.node_id,
                            "to_node": self.state.current_node.node_id,
                            "message": self.state.handoff_message
                        }

                except asyncio.TimeoutError:
                    logger.exception(
                        "node=<%s>, timeout=<%s>s | node execution timed out after timeout",
                        current_node.node_id,
                        self.node_timeout,
                    )
                    self.state.completion_status = "FAILED"
                    yield {
                        "type": "node_timeout",
                        "node_id": current_node.node_id,
                        "timeout": self.node_timeout
                    }
                    break

                except Exception as e:
                    logger.exception("node=<%s> | node execution failed", current_node.node_id)
                    self.state.completion_status = "FAILED"
                    yield {
                        "type": "node_error",
                        "node_id": current_node.node_id,
                        "error": str(e)
                    }
                    break

        except Exception as e:
            logger.exception("streaming swarm execution failed")
            self.state.completion_status = "FAILED"
            yield {
                "type": "error",
                "error": str(e)
            }

    async def _stream_execute_node(self, node, task: str | list[ContentBlock]) -> AsyncIterator[Any]:
        """Stream individual node execution"""
        start_time = time.time()
        node_name = node.node_id

        try:
            # Prepare context for node
            context_text = self._build_node_input(node)
            node_input = [ContentBlock(text=f"Context:\n{context_text}\n\n")]

            # Clear handoff message after it's been included in context
            self.state.handoff_message = None

            if not isinstance(task, str):
                # Include additional ContentBlocks in node input
                node_input = node_input + task

            # Execute node with streaming
            result = None
            node.reset_executor_state()
            
            # Stream the agent execution
            events = node.executor.stream_async(node_input)
            current_tool_use = None
            
            async for event in events:
                # Process different event types and yield specific events
                if "data" in event:
                    # Real-time text streaming
                    yield {
                        "type": "node_text_stream",
                        "node_id": node_name,
                        "text": event["data"]
                    }
                    
                elif "current_tool_use" in event:
                    # Store current tool use for later processing
                    current_tool_use = event["current_tool_use"]
                    
                elif "message" in event:
                    # Message event (including tool results)
                    message = event["message"]
                    if "content" in message:
                        content = message["content"]
                        for content_item in content:
                            if "toolResult" in content_item:
                                # Tool result event
                                tool_result = content_item["toolResult"]
                                tool_use_id = tool_result.get("toolUseId", "")
                                
                                # First yield the complete tool usage info if we have it
                                if current_tool_use and current_tool_use.get("toolUseId") == tool_use_id:
                                    yield {
                                        "type": "node_tool_use",
                                        "node_id": node_name,
                                        "tool_name": current_tool_use.get("name", ""),
                                        "tool_input": current_tool_use.get("input", ""),
                                        "tool_use_id": tool_use_id
                                    }
                                    current_tool_use = None  # Clear after use
                                
                                # Then yield the tool result
                                tool_content = tool_result.get("content", [])
                                for tool_content_item in tool_content:
                                    if "text" in tool_content_item:
                                        yield {
                                            "type": "node_tool_result",
                                            "node_id": node_name,
                                            "tool_result": tool_content_item["text"],
                                            "tool_use_id": tool_use_id
                                        }
                            elif "text" in content_item:
                                # Regular text content
                                text_content = content_item["text"]
                                if text_content.strip():
                                    yield {
                                        "type": "node_text",
                                        "node_id": node_name,
                                        "text": text_content
                                    }
                    
                elif "result" in event:
                    # Final result event
                    result = event["result"]
                    yield {
                        "type": "node_result",
                        "node_id": node_name,
                        "result": result
                    }

            execution_time = round((time.time() - start_time) * 1000)

            # Create simple result object
            class SimpleNodeResult:
                def __init__(self, result, execution_time, status):
                    self.result = result
                    self.execution_time = execution_time
                    self.status = status
                    self.accumulated_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
                    self.accumulated_metrics = {"latencyMs": execution_time}

            node_result = SimpleNodeResult(result, execution_time, "COMPLETED")

            # Store result in state
            self.state.results[node_name] = node_result

            # Accumulate metrics
            self._accumulate_metrics(node_result)

            # Yield node completion event
            yield {
                "type": "node_complete",
                "node_id": node_name,
                "execution_time": execution_time,
                "result": result
            }

        except Exception as e:
            execution_time = round((time.time() - start_time) * 1000)
            logger.exception("node=<%s> | node execution failed", node_name)

            # Create a simple result for the failed node
            class SimpleNodeResult:
                def __init__(self, result, execution_time, status):
                    self.result = result
                    self.execution_time = execution_time
                    self.status = status
                    self.accumulated_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
                    self.accumulated_metrics = {"latencyMs": execution_time}

            node_result = SimpleNodeResult(e, execution_time, "FAILED")

            # Store result in state
            self.state.results[node_name] = node_result

            yield {
                "type": "node_error",
                "node_id": node_name,
                "error": str(e),
                "execution_time": execution_time
            }
            raise

    def _build_node_input(self, target_node) -> str:
        """Build input text for a node based on shared context and handoffs."""
        context_info = {
            "task": self.state.task,
            "node_history": [node.node_id for node in self.state.node_history],
            "shared_context": getattr(self.state.shared_context, 'context', {}),
        }
        context_text = ""

        # Include handoff message prominently at the top if present
        if self.state.handoff_message:
            context_text += f"Handoff Message: {self.state.handoff_message}\n\n"

        # Include task information if available
        if "task" in context_info:
            task = context_info.get("task")
            if isinstance(task, str):
                context_text += f"User Request: {task}\n\n"
            elif isinstance(task, list):
                context_text += "User Request: Multi-modal task\n\n"

        # Include detailed node history
        if context_info.get("node_history"):
            context_text += f"Previous agents who worked on this: {' → '.join(context_info['node_history'])}\n\n"

        # Include actual shared context, not just a mention
        shared_context = context_info.get("shared_context", {})
        if shared_context:
            context_text += "Shared knowledge from previous agents:\n"
            for node_name, context in shared_context.items():
                if context:  # Only include if node has contributed context
                    context_text += f"• {node_name}: {context}\n"
            context_text += "\n"

        # Include available nodes with descriptions if available
        other_nodes = [node_id for node_id in self.nodes.keys() if node_id != target_node.node_id]
        if other_nodes:
            context_text += "Other agents available for collaboration:\n"
            for node_id in other_nodes:
                node = self.nodes.get(node_id)
                context_text += f"Agent name: {node_id}."
                if node and hasattr(node.executor, "description") and node.executor.description:
                    context_text += f" Agent description: {node.executor.description}"
                context_text += "\n"
            context_text += "\n"

        context_text += (
            "You have access to swarm coordination tools if you need help from other agents. "
            "If you don't hand off to another agent, the swarm will consider the task complete."
        )

        return context_text

    def _accumulate_metrics(self, node_result) -> None:
        """Accumulate metrics from a node result."""
        self.state.accumulated_usage["inputTokens"] += node_result.accumulated_usage.get("inputTokens", 0)
        self.state.accumulated_usage["outputTokens"] += node_result.accumulated_usage.get("outputTokens", 0)
        self.state.accumulated_usage["totalTokens"] += node_result.accumulated_usage.get("totalTokens", 0)
        self.state.accumulated_metrics["latencyMs"] += node_result.accumulated_metrics.get("latencyMs", 0)

    def _handle_handoff(self, target_node, message: str, context: dict[str, Any]) -> None:
        """Handle handoff to another agent."""
        # If task is already completed, don't allow further handoffs
        if self.state.completion_status != "EXECUTING":
            logger.debug(
                "task_status=<%s> | ignoring handoff request - task already completed",
                self.state.completion_status,
            )
            return

        # Update swarm state
        previous_agent = self.state.current_node
        self.state.current_node = target_node

        # Store handoff message for the target agent
        self.state.handoff_message = message

        # Store handoff context as shared context
        if context:
            for key, value in context.items():
                self.state.shared_context.add_context(previous_agent, key, value)

        logger.debug(
            "from_node=<%s>, to_node=<%s> | handed off from agent to agent",
            previous_agent.node_id,
            target_node.node_id,
        )

    def _build_result(self):
        """Build swarm result from current state."""
        class SimpleSwarmResult:
            def __init__(self, status, results, accumulated_usage, accumulated_metrics, execution_count, execution_time, node_history):
                self.status = status
                self.results = results
                self.accumulated_usage = accumulated_usage
                self.accumulated_metrics = accumulated_metrics
                self.execution_count = execution_count
                self.execution_time = execution_time
                self.node_history = node_history
        
        return SimpleSwarmResult(
            status=self.state.completion_status,
            results=self.state.results,
            accumulated_usage=self.state.accumulated_usage,
            accumulated_metrics=self.state.accumulated_metrics,
            execution_count=len(self.state.node_history),
            execution_time=self.state.execution_time,
            node_history=self.state.node_history,
        )


async def stream_swarm_execution(swarm, question, containers):
    """Stream swarm execution events in real-time"""
    global index
    final_result = ""
    current_node = ""
    # Track tool usage to avoid duplicate notifications
    tool_usage_cache = {}
    # Use a single notification container for streaming updates
    streaming_container_index = None
    # Accumulate all streaming content
    streaming_content = ""
    
    try:
        # Stream swarm events
        async for event in swarm.stream_async(question):
            event_type = event.get("type", "")
            
            if event_type == "swarm_start":
                add_notification(containers, f"🚀 Swarm started with {event.get('current_node', 'unknown')}")
                
            elif event_type == "node_start":
                current_node = event.get("node_id", "")
                iteration = event.get("iteration", 0)
                add_notification(containers, f"🔄 {current_node} starting (Step {iteration})")
                
            elif event_type == "node_text_stream":
                # Real-time text streaming from agent
                node_id = event.get("node_id", "")
                text = event.get("text", "")
                
                # Initialize streaming container if needed
                if streaming_container_index is None:
                    streaming_container_index = index
                    index += 1
                
                # Simple approach: just accumulate text without complex parsing
                # This avoids issues with incomplete code blocks during streaming
                streaming_content += text
                
                # Check if we have complete code blocks to avoid partial parsing
                code_block_count = streaming_content.count("```")
                if code_block_count % 2 == 1:
                    # Incomplete code block, don't process yet
                    continue
                
                # Update the container with accumulated content
                if containers is not None:
                    # Clean and format content for proper display
                    formatted_content = streaming_content.strip()
                    # Convert HTML <br> tags back to newlines for proper display
                    formatted_content = formatted_content.replace('<br>', '\n')
                    # Clean up excessive whitespace
                    formatted_content = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_content)
                    containers['notification'][streaming_container_index].info(formatted_content)
                
            elif event_type == "node_tool_use":
                # Tool usage event
                node_id = event.get("node_id", "")
                tool_name = event.get("tool_name", "")
                tool_input = event.get("tool_input", "")
                tool_use_id = event.get("tool_use_id", "")
                
                # Create a cache key for this tool usage (use tool_use_id to track complete tool usage)
                cache_key = f"{node_id}_{tool_name}_{tool_use_id}"
                
                # Only show tool usage if it's new or different
                if cache_key not in tool_usage_cache:
                    tool_usage_cache[cache_key] = True
                    
                    # Initialize streaming container if needed
                    if streaming_container_index is None:
                        streaming_container_index = index
                        index += 1
                    
                    # Add tool usage info to streaming content
                    tool_info = f"🔧 {node_id} using tool: {tool_name}"
                    if tool_input:
                        if isinstance(tool_input, dict):
                            if tool_name == "handoff_to_agent":
                                agent_name = tool_input.get("agent_name", "")
                                message = tool_input.get("message", "")
                                context = tool_input.get("context", {})
                                
                                tool_info += f"\n📝 Handing off to: {agent_name}"
                                if message:
                                    # Show only the message, not the full JSON
                                    short_message = message[:150] + "..." if len(message) > 150 else message
                                    tool_info += f"\n📝 Message: {short_message}"
                                if context:
                                    # Show only key context info
                                    if isinstance(context, dict):
                                        context_summary = ", ".join([f"{k}: {v}" for k, v in context.items() if isinstance(v, (str, int, bool))])
                                        if context_summary:
                                            tool_info += f"\n📝 Context: {context_summary[:100]}..."
                                    else:
                                        context_str = str(context)[:100] + "..." if len(str(context)) > 100 else str(context)
                                        tool_info += f"\n📝 Context: {context_str}"
                            else:
                                # For other tools, show key-value pairs
                                input_str = ", ".join([f"{k}: {v}" for k, v in tool_input.items()])
                                tool_info += f"\n📝 Tool input: {input_str}"
                        else:
                            input_str = str(tool_input)
                            tool_info += f"\n📝 Tool input: {input_str}"
                    
                    # Accumulate content
                    streaming_content += f"\n{tool_info}"
                    
                    # Update the container with accumulated content
                    if containers is not None:
                        # Clean and format content for proper display
                        formatted_content = streaming_content.strip()
                        # Convert HTML <br> tags back to newlines for proper display
                        formatted_content = formatted_content.replace('<br>', '\n')
                        # Clean up excessive whitespace
                        formatted_content = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_content)
                        containers['notification'][streaming_container_index].info(formatted_content)
                    
            elif event_type == "node_tool_result":
                # Tool result event
                node_id = event.get("node_id", "")
                tool_result = event.get("tool_result", "")
                tool_use_id = event.get("tool_use_id", "")
                
                # Create a cache key for this tool result
                result_cache_key = f"{node_id}_result_{tool_use_id}_{tool_result}"
                
                # Only show tool result if it's new or different
                if result_cache_key not in tool_usage_cache:
                    tool_usage_cache[result_cache_key] = True
                    
                    # Initialize streaming container if needed
                    if streaming_container_index is None:
                        streaming_container_index = index
                        index += 1
                    
                    # Add tool result to streaming content
                    tool_result_info = f"📤 Tool result from {node_id}: {tool_result}"
                    streaming_content += f"\n{tool_result_info}"
                    
                    # Update the container with accumulated content
                    if containers is not None:
                        # Clean and format content for proper HTML display
                        formatted_content = streaming_content.strip()
                        # Replace multiple newlines with single <br> and single newlines with <br>
                        formatted_content = re.sub(r'\n\s*\n', '<br><br>', formatted_content)
                        formatted_content = formatted_content.replace('\n', '<br>')
                        # Remove excessive whitespace
                        formatted_content = re.sub(r'\s+', ' ', formatted_content)
                        containers['notification'][streaming_container_index].info(formatted_content)
                
            elif event_type == "node_text":
                # Regular text content from agent
                node_id = event.get("node_id", "")
                text = event.get("text", "")
                if text.strip():
                    # Create a cache key for this text
                    text_cache_key = f"{node_id}_text_{text}"
                    
                    # Only show text if it's new or different
                    if text_cache_key not in tool_usage_cache:
                        tool_usage_cache[text_cache_key] = True
                        
                        # Initialize streaming container if needed
                        if streaming_container_index is None:
                            streaming_container_index = index
                            index += 1
                        
                        # Simple approach: just add text without complex parsing
                        text_info = f"💬 {node_id}: {text}"
                        streaming_content += f"\n{text_info}"
                        
                        # Update the container with accumulated content
                        if containers is not None:
                            # Clean and format content for proper display
                            formatted_content = streaming_content.strip()
                            # Convert HTML <br> tags back to newlines for proper display
                            formatted_content = formatted_content.replace('<br>', '\n')
                            # Clean up excessive whitespace
                            formatted_content = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_content)
                            containers['notification'][streaming_container_index].info(formatted_content)
                    
            elif event_type == "node_result":
                # Final result from this node
                node_id = event.get("node_id", "")
                result = event.get("result")
                if hasattr(result, 'message') and result.message:
                    content = result.message.get("content", [])
                    if content and len(content) > 0:
                        text_content = content[0].get("text", "")
                        if text_content:
                            # Initialize streaming container if needed
                            if streaming_container_index is None:
                                streaming_container_index = index
                                index += 1
                            
                            # Add result to streaming content
                            result_info = f"✅ {node_id} completed: {text_content}"
                            streaming_content += f"\n{result_info}"
                            
                            # Update the container with accumulated content
                            if containers is not None:
                                # Clean and format content for proper display
                                formatted_content = streaming_content.strip()
                                # Convert HTML <br> tags back to newlines for proper display
                                formatted_content = formatted_content.replace('<br>', '\n')
                                # Clean up excessive whitespace
                                formatted_content = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_content)
                                containers['notification'][streaming_container_index].info(formatted_content)
                            
                            final_result = text_content
                                
            elif event_type == "handoff":
                from_node = event.get("from_node", "")
                to_node = event.get("to_node", "")
                message = event.get("message", "")
                
                # Initialize streaming container if needed
                if streaming_container_index is None:
                    streaming_container_index = index
                    index += 1
                
                # Add handoff info to streaming content
                handoff_info = f"🔄 Handoff: {from_node} → {to_node}"
                if message:
                    handoff_info += f"\n💬 Handoff message: {message}"
                    if len(message) > 100:
                        handoff_info += f"\n📋 Full handoff context: {message}"
                
                streaming_content += f"\n{handoff_info}"
                
                # Update the container with accumulated content
                if containers is not None:
                    # Clean and format content for proper display
                    formatted_content = streaming_content.strip()
                    # Convert HTML <br> tags back to newlines for proper display
                    formatted_content = formatted_content.replace('<br>', '\n')
                    # Clean up excessive whitespace
                    formatted_content = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_content)
                    containers['notification'][streaming_container_index].info(formatted_content)
                    
            elif event_type == "node_complete":
                node_id = event.get("node_id", "")
                execution_time = event.get("execution_time", 0)
                
                # Initialize streaming container if needed
                if streaming_container_index is None:
                    streaming_container_index = index
                    index += 1
                
                # Add completion info to streaming content
                completion_info = f"✅ {node_id} completed in {execution_time}ms"
                streaming_content += f"\n{completion_info}"
                
                # Update the container with accumulated content
                if containers is not None:
                    # Clean and format content for proper display
                    formatted_content = streaming_content.strip()
                    # Convert HTML <br> tags back to newlines for proper display
                    formatted_content = formatted_content.replace('<br>', '\n')
                    # Clean up excessive whitespace
                    formatted_content = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_content)
                    containers['notification'][streaming_container_index].info(formatted_content)
                
            elif event_type == "swarm_complete":
                result = event.get("result")
                if result:
                    add_notification(containers, f"🎉 Swarm execution completed!")
                    add_notification(containers, f"📊 Total execution time: {result.execution_time}ms")
                    add_notification(containers, f"📈 Total tokens: {result.accumulated_usage.get('totalTokens', 0)}")
                    
            elif event_type == "error":
                error_msg = event.get("error", "Unknown error")
                add_notification(containers, f"❌ Error: {error_msg}")
                
            elif event_type == "node_timeout":
                node_id = event.get("node_id", "")
                timeout = event.get("timeout", 0)
                add_notification(containers, f"⏰ {node_id} timed out after {timeout}s")
                
            elif event_type == "node_error":
                node_id = event.get("node_id", "")
                error_msg = event.get("error", "Unknown error")
                add_notification(containers, f"❌ {node_id} error: {error_msg}")
                
    except Exception as e:
        logger.error(f"Streaming swarm execution failed: {str(e)}")
        add_notification(containers, f"❌ Streaming error: {str(e)}")
        final_result = f"Error: {str(e)}"
    
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

    # Create a streaming swarm with these agents
    swarm = StreamingSwarm(
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
    
    # Use stream_async for real-time streaming updates
    try:
        add_notification(containers, "Starting streaming swarm execution...")
        
        # Stream swarm execution events
        final_result = await stream_swarm_execution(swarm, question, containers)
        
    except Exception as e:
        logger.error(f"Swarm execution failed: {str(e)}")
        final_result = f"Error: {str(e)}"

    if containers is not None:
        containers['notification'][index].markdown(final_result)

    logger.info(f"Final result: {final_result}")

    return final_result
