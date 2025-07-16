# Strands Agent

[Strands agent](https://strandsagents.com/0.1.x/)는 AI agent 구축 및 실행을 위해 설계된 오픈소스 SDK입니다. 계획(planning), 사고 연결(chaining thoughts), 도구 호출, Reflection과 같은 agent 기능을 쉽게 활용할 수 있습니다. 이를 통해 LLM model과 tool을 연결하며, 모델의 추론 능력을 이용하여 도구를 계획하고 실행합니다. 현재 Amazon Bedrock, Anthropic, Meta의 모델을 지원하며, Accenture, Anthropic, Meta와 같은 기업들이 참여하고 있습니다. 

Strands agent는 아래와 같은 [Agent Loop](https://strandsagents.com/0.1.x/user-guide/concepts/agents/agent-loop/)가지고 있으므로, 적절한 tool을 선택하여 실행하고, reasoning을 통해 반복적으로 필요한 동작을 수행합니다. 

![image](https://github.com/user-attachments/assets/6f641574-9d0b-4542-b87f-98d7c2715e09)

Tool들을 아래와 같이 병렬로 처리할 수 있습니다.

```python
agent = Agent(
    max_parallel_tools=4  
)
```

## Strands Agent 활용 방법

### Model 설정

모델 설정과 관련된 paramter는 아래와 같습니다.

- max_tokens: Maximum number of tokens to generate in the response
- model_id: The Bedrock model ID (e.g., "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
- stop_sequences: List of sequences that will stop generation when encountered
- temperature: Controls randomness in generation (higher = more random)
- top_p: Controls diversity via nucleus sampling (alternative to temperature)
- cache_prompt: Cache point type for the system prompt
- cache_tools: Cache point type for tools
- guardrail_id: ID of the guardrail to apply
- guardrail_trace: Guardrail trace mode. Defaults to enabled.
- guardrail_version: Version of the guardrail to apply
- guardrail_stream_processing_mode: The guardrail processing mode
- guardrail_redact_input: Flag to redact input if a guardrail is triggered. Defaults to True.
- guardrail_redact_input_message: If a Bedrock Input guardrail triggers, replace the input with this message.
- guardrail_redact_output: Flag to redact output if guardrail is triggered. Defaults to False.
- guardrail_redact_output_message: If a Bedrock Output guardrail triggers, replace output with this message.

Reasoning mode와 standard mode의 설정이 다르므로 아래와 같이 설정합니다.

```python
def get_model():
    profile = models[0]
    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 

    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k

    maxReasoningOutputTokens=64000
    thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

    if reasoning_mode=='Enable':
        model = BedrockModel(
            boto_client_config=Config(
               read_timeout=900,
               connect_timeout=900,
               retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=64000,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 1,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            },
        )
    else:
        model = BedrockModel(
            boto_client_config=Config(
               read_timeout=900,
               connect_timeout=900,
               retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=maxOutputTokens,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 0.1,
            top_p = 0.9,
            additional_request_fields={
                "thinking": {
                    "type": "disabled"
                }
            }
        )
    return model
```

### Agent의 실행

아래와 같이 system prompt, model, tool 정보를 가지고 agent를 생성합니다.

```python
def create_agent():
    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )

    model = get_model()

    agent = Agent(
        model=model,
        system_prompt=system,
        tools=[    
            calculator, 
            current_time,
            use_aws    
        ],
    )
    return agent
```

Agent는 stream으로 결과를 주므로, 아래와 같이 event에서 "data"만을 추출한 후에 full_response로 저장한 후에 markdown으로 표시합니다. 

```python
def run_strands_agent(question, history_mode, st):
    agent = create_agent()
    
    message_placeholder = st.empty()
    full_response = ""

    async def process_streaming_response():
        nonlocal full_response
        agent_stream = agent.stream_async(question)
        async for event in agent_stream:
            if "data" in event:
                full_response += event["data"]
                message_placeholder.markdown(full_response)

    asyncio.run(process_streaming_response())

    return full_response
```

### 대화 이력의 활용

대화 내용을 이용해 대화를 이어나가고자 할 경우에 아래와 같이 SlidingWindowConversationManager을 이용해서 window_size만큼 이전 대화를 가져와 활용할 수 있습니다. 상세한 코드는 [chat.py](./application/chat.py)을 참조합니다.

```python
from strands.agent.conversation_manager import SlidingWindowConversationManager

conversation_manager = SlidingWindowConversationManager(
    window_size=10,  
)

agent = Agent(
    model=model,
    system_prompt=system,
    tools=[    
        calculator, 
        current_time,
        use_aws    
    ],
    conversation_manager=conversation_manager
)
```

### MCP 활용

아래와 같이 MCPClient로 stdio_mcp_client을 지정한 후에 list_tools_sync을 이용해 tool 정보를 추출합니다. MCP tool은 strands tool과 함께 아래처럼 사용할 수 있습니다.

```python
from strands.tools.mcp import MCPClient
from strands_tools import calculator, current_time, use_aws

stdio_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"])
))

with stdio_mcp_client as client:
    aws_documentation_tools = client.list_tools_sync()
    logger.info(f"aws_documentation_tools: {aws_documentation_tools}")

    tools=[    
        calculator, 
        current_time,
        use_aws
    ]

    tools.extend(aws_documentation_tools)

    if history_mode == "Enable":
        logger.info("history_mode: Enable")
        agent = Agent(
            model=model,
            system_prompt=system,
            tools=tools,
            conversation_manager=conversation_manager
        )
    else:
        logger.info("history_mode: Disable")
        agent = Agent(
            model=model,
            system_prompt=system,
            tools=tools
        )
```

또한, wikipedia 검색을 위한 MCP server의 예는 아래와 같습니다. 상세한 코드는 [mcp_server_wikipedia.py](./application/mcp_server_wikipedia.py)을 참조합니다.

```python
from mcp.server.fastmcp import FastMCP
import wikipedia
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("rag")

mcp = FastMCP(
    "Wikipedia",
    dependencies=["wikipedia"],
)

@mcp.tool()
def search(query: str):
    logger.info(f"Searching Wikipedia for: {query}")
    
    return wikipedia.search(query)

@mcp.tool()
def summary(query: str):
    return wikipedia.summary(query)

@mcp.tool()
def page(query: str):
    return wikipedia.page(query)

@mcp.tool()
def random():
    return wikipedia.random()

@mcp.tool()
def set_lang(lang: str):
    wikipedia.set_lang(lang)
    return f"Language set to {lang}"

if __name__ == "__main__":
    mcp.run()
```

### 동적으로 MCP Server를 binding하기

MCP Server를 동적으로 관리하기 위하여 MCPClientManager를 정의합니다. add_client는 MCP 서버의 name, command, args, env로 MCP Client를 정의합니다. 

```python
class MCPClientManager:
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        
    def add_client(self, name: str, command: str, args: List[str], env: dict[str, str] = {}) -> None:
        """Add a new MCP client"""
        self.clients[name] = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=command, args=args, env=env
            )
        ))
    
    def remove_client(self, name: str) -> None:
        """Remove an MCP client"""
        if name in self.clients:
            del self.clients[name]
    
    @contextmanager
    def get_active_clients(self, active_clients: List[str]):
        """Manage active clients context"""
        active_contexts = []
        for client_name in active_clients:
            if client_name in self.clients:
                active_contexts.append(self.clients[client_name])

        if active_contexts:
            with contextlib.ExitStack() as stack:
                for client in active_contexts:
                    stack.enter_context(client)
                yield
        else:
            yield

# Initialize MCP client manager
mcp_manager = MCPClientManager()
```

Streamlit으로 구현한 [app.py](./application/app.py)에서 tool들을 선택하면 mcp_tools를 얻을 수 있습니다. 이후 아래와 같이 agent 생성시에 active client으로 부터 tool list를 가져와서 tools로 활용합니다.

```python
tools = []
for mcp_tool in mcp_tools:
    logger.info(f"mcp_tool: {mcp_tool}")
    with mcp_manager.get_active_clients([mcp_tool]) as _:
        if mcp_tool in mcp_manager.clients:
            client = mcp_manager.clients[mcp_tool]
            mcp_tools_list = client.list_tools_sync()
            tools.extend(mcp_tools_list)
```

tools 정보는 아래와 같이 agent 생성시 활용됩니다.

```python
agent = Agent(
    model=model,
    system_prompt=system,
    tools=tools
)
```

생성된 agent는 아래와 같이 mcp_manager를 이용해 실행합니다.

```python
with mcp_manager.get_active_clients(mcp_tools) as _:
    agent_stream = agent.stream_async(question)
    
    tool_name = ""
    async for event in agent_stream:
        if "message" in event:
            message = event["message"]
            for content in message["content"]:                
                if "text" in content:
                    final_response = content["text"]
```

### Streamlit에 맞게 출력문 조정하기

Agent를 아래와 같이 실행하여 agent_stream을 얻습니다.

```python
with mcp_manager.get_active_clients(mcp_servers) as _:
    agent_stream = agent.stream_async(question)
```

사용자 경험을 위해서는 stream형태로 출력을 얻을 수 있어야 합니다. 이는 아래와 같이 agent_stream에서 event를 꺼낸후 "data"에서 추출하여 아래와 같이 current_response에 stream 결과를 모아서 보여줍니다.

```python
async for event in agent_stream:
    if "data" in event:
        text_data = event["data"]
        current_response += text_data

        containers["notification"][index].markdown(current_response)
```

Strands agent는 multi step reasoning을 통해 여러번 결과가 나옵니다. 최종 결과를 얻기 위해 아래와 같이 message의 content에서 text를 추출하여 마지막만을 추출합니다. 또한 tool마다 reference가 다르므로 아래와 같이 tool content의 text에서 reference를 추출합니다.  

```python
if "message" in event:
    message = event["message"]
    for msg_content in message["content"]:                
        result = msg_content["text"]
        current_response = ""

        tool_content = msg_content["toolResult"]["content"]
        for content in tool_content:
            content, urls, refs = get_tool_info(tool_name, content["text"])
            if refs:
                for r in refs:
                    references.append(r)
```

generate_image_with_colors라는 tool의 최종 이미지 경로는 아래와 같이 event_loop_metrics에서 추출합하여 image_urls로 활용합니다.

```python
if "event_loop_metrics" in event and \
    hasattr(event["event_loop_metrics"], "tool_metrics") and \
    "generate_image_with_colors" in event["event_loop_metrics"].tool_metrics:
    tool_info = event["event_loop_metrics"].tool_metrics["generate_image_with_colors"].tool
    if "input" in tool_info and "filename" in tool_info["input"]:
        fname = tool_info["input"]["filename"]
        if fname:
            url = f"{path}/{s3_image_prefix}/{parse.quote(fname)}.png"
            if url not in image_urls:
                image_urls.append(url)
```

## Multi Agent

### Supervisor

[Agents as Tools](https://strandsagents.com/latest/user-guide/concepts/multi-agent/agents-as-tools/#implementing-agents-as-tools-with-strands-agents-sdk)와 같이 orchestrator agent를 이용해 research_assistant, product_recommendation_assistant, trip_planning_assistant와 같은 여러 agents를 이용할 수 있습니다. [agent-as-tools](https://github.com/strands-agents/samples/tree/main/01-tutorials/02-multi-agent-systems/01-agent-as-tool)와 같이 supervisor형태의 multi agent은 tool에 속한 agent을 이용해 구현할 수 있습니다.

<img width="800" height="287" alt="image" src="https://github.com/user-attachments/assets/bf983b09-912d-456e-a774-1861fc873fba" />

[strands_supervisor.py](./application/strands_supervisor.py)와 같이 supervisor를 위한 orchestration agent를 생성할 수 있습니다. 이 agent는 research_assistant, product_recommendation_assistant, trip_planning_assistant의 agent로 만들어진 tool을 가지고 있습니다.

```python
orchestrator = Agent(
    model=strands_agent.get_model(),
    system_prompt=MAIN_SYSTEM_PROMPT,
    tools=[
        research_assistant,
        product_recommendation_assistant,
        trip_planning_assistant,
        file_write,
    ],
)
agent_stream = orchestrator.stream_async(question)
result = await show_streams(agent_stream, containers)
```

여기서 trip_planning_assistant는 아래와 같이 travel_agent라는 agent를 가지고 있습니다.

```python
@tool
async def trip_planning_assistant(query: str) -> str:
    """
    Create travel itineraries and provide travel advice.
    Args:
        query: A travel planning request with destination and preferences
    Returns:
        A detailed travel itinerary or travel advice
    """
    travel_agent = Agent(
        model=strands_agent.get_model(),
        system_prompt=TRIP_PLANNING_ASSISTANT_PROMPT,
    )        
    agent_stream = travel_agent.stream_async(query)
    result = await show_streams(agent_stream, containers)

    return result
```


### Swarm

[Multi-Agent Systems and Swarm Intelligence](https://strandsagents.com/latest/user-guide/concepts/multi-agent/swarm/)와 같이 Agent들이 서로 협조하면서 복잡한 문제를 해결 할 수 있습니다. 

#### Mesh Swarm

[Mesh Swarm Architecture](https://strandsagents.com/latest/user-guide/concepts/multi-agent/swarm/#mesh-swarm-architecture)와 같이 여러 agent들간의 협업을 수행하 수 있습니다. Research agent는 논리적인 답변을, creative agent는 흥미로운 답변을 제공합니다. 이때, critical agent로 두 agent들의 개선점을 도출한 후에, summarizer agent로 최적의 답변을 구할 수 있습니다.

<img width="700" alt="swarm" src="https://github.com/user-attachments/assets/b2d400b5-87f2-4a1a-9e28-877e107834c2" />


이를 구현하는 방법은 [strands_swarm.py](./application/strands_swarm.py)을 참조합니다. 먼저 질문을 research, creative, critical agent에 전달해 답변을 구하고, 결과를 다른 agent들에 전달합니다.

```python
result = research_agent.stream_async(question)
research_result = await show_streams(result, containers)

result = creative_agent.stream_async(question)
creative_result = await show_streams(result, containers)

result = critical_agent.stream_async(question)
critical_result = await show_streams(result, containers)

creative_messages.append(f"From Research Agent: {research_result}")
critical_messages.append(f"From Research Agent: {research_result}")
summarizer_messages.append(f"From Research Agent: {research_result}")

research_messages.append(f"From Creative Agent: {creative_result}")
critical_messages.append(f"From Creative Agent: {creative_result}")
summarizer_messages.append(f"From Creative Agent: {creative_result}")

research_messages.append(f"From Critical Agent: {critical_result}")
creative_messages.append(f"From Critical Agent: {critical_result}")
summarizer_messages.append(f"From Critical Agent: {critical_result}")
```

결과를 refine하고 얻어진 결과를 summarizer agent에 전달합니다.

```python
result = research_agent.stream_async(research_prompt)
refined_research = await show_streams(result, containers)

result = creative_agent.stream_async(creative_prompt)
refined_creative = await show_streams(result, containers)

result = critical_agent.stream_async(critical_prompt)
refined_critical = await show_streams(result, containers)

summarizer_messages.append(f"From Research Agent (Phase 2): {refined_research}")
summarizer_messages.append(f"From Creative Agent (Phase 2): {refined_creative}")
summarizer_messages.append(f"From Critical Agent (Phase 2): {refined_critical}")
```

이후 아래와 같이 요약합니다.

```python
summarizer_prompt = f"""
Original query: {question}

Please synthesize the following inputs from all agents into a comprehensive final solution:

{"\n\n".join(summarizer_messages)}

Create a well-structured final answer that incorporates the research findings, 
creative ideas, and addresses the critical feedback.
"""

result = summarizer_agent.stream_async(summarizer_prompt)
final_solution = await show_streams(result, containers)
```

#### Swarm Tool

[Creating Swarm of agents using Strands Agents](https://github.com/strands-agents/samples/blob/main/01-tutorials/02-multi-agent-systems/02-swarm-agent/swarm.ipynb)에서 strands agent에서 swarm을 사용할 수 있도록 tool을 제공하고 있습니다. 이때 agent에서 설정할 수 있는 협업 옵션은 아래와 같습니다.

- Collaborative: Agents build upon others' insights and seek consensus
- Competitive: Agents develop independent solutions and unique perspectives
- Hybrid: Balances cooperation with independent exploration

협업하는 swarm agent들로부터 얻어진 결과를 summarized agent로 정리하여 답변합니다. 아래는 swarm tool을 사용할때의 diagram입니다. 여기서 swarm agent의 숫자는 swarm_size로 조정합니다.

<img width="600" alt="swarm_tool" src="https://github.com/user-attachments/assets/fd9b69f4-3d85-4dae-ab7f-347ef207f862" />

[strands_swarm_tool.py](./application/strands_swarm_tool.py)와 같이 strands agent를 이용해 swarm 형태의 multi agent를 구현하고, 이를 통해 복잡한 문제를 풀 수 있습니다.

```python
from strands_tools import swarm
system_prompt = (
    "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
    "모르는 질문을 받으면 솔직히 모른다고 말합니다."
)

agent = Agent(
    model=strands_agent.get_model(),
    system_prompt=system_prompt,
    tools=[swarm]
)

result = agent.tool.swarm(
    task=question,
    swarm_size=3,
    coordination_pattern="collaborative"
)    
logger.info(f"result of swarm: {result}")
```

Swarm tool로 얻어진 결과는 아래와 같이 summarize agent로 요약합니다.

```python
summarizer_prompt = f"""
질문: <question>{question}</question>

아래 에이전트들의 생각을 종합하여 최종 답변을 생성하세요. 
<opinion>{"\n\n".join(messages)}</opinion>
"""    

model = strands_agent.get_model()
summarizer_agent = Agent(
    model=model,
    system_prompt=summarizer_prompt,
)    
agent_stream = summarizer_agent.stream_async(question)
result = await show_streams(agent_stream, containers)
```


### Workflow

[Agent Workflows](https://strandsagents.com/latest/user-guide/concepts/multi-agent/workflow/#implementing-workflow-architectures)을 이용하면 간단한 workflow를 손쉽게 구현할 수 있습니다.

<img width="614" height="73" alt="image" src="https://github.com/user-attachments/assets/3473d42f-657a-4056-8eb7-ced1605916b8" />

[strands_workflow.py](./application/strands_workflow.py)에서는 아래와 같이 researcher, analyst, writer를 통해 좀더 심화된 보고서를 생성할 수 있습니다.

```python
async def run_workflow(question, containers):
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

    # Step 2: Analysis
    add_notification(containers, f"분석: {research_result}")
    analysis = f"다음을 분석해서 필요한 데이터를 추가하고 이해하기 쉽게 분석하세요. <research>{research_result}</research>"
    analysis_stream = analyst.stream_async(analysis)
    analysis_result = await show_streams(analysis_stream, containers)    

    # Step 3: Report writing
    add_notification(containers, f"보고서: {analysis_result}")
    report = f"다음의 내용을 참조하여 상세한 보고서를 작성하세요. <subject>{analysis_result}</subject>"
    report_stream = writer.stream_async(report)
    report_result = await show_streams(report_stream, containers)    

    return report_result
```

### Graph

[Agent Graphs](https://strandsagents.com/latest/user-guide/concepts/multi-agent/graph/#implementing-agent-graphs-with-strands)와 같이 다단계로 된 복잡한 Graph를 구현할 수 있습니다. 이때의 agent들의 구성도는 아래와 같습니다.

<img width="386" height="409" alt="image" src="https://github.com/user-attachments/assets/a6495615-8357-4ae6-8444-cf33ff714047" />

[strands_graph.py](./application/strands_graph.py)와 같이 구현할 수 있습니다. 여기서 graph의 시작은 coordinator입니다. 이 agent는 economic_department, technical_analysis, social_analysis을 가지고 있습니다.

```python
coordinator = Agent(
    system_prompt=COORDINATOR_SYSTEM_PROMPT,
    tools=[economic_department, technical_analysis, social_analysis]
)
agent_stream = coordinator.stream_async(f"Provide a comprehensive analysis of: {question}")
```

여기서 economic_department는 아래와 같이 tool로 구현됩니다. 이 agent도 market_research, financial_analysis를 tool로 가지고 있습니다.

```python
@tool
async def economic_department(query: str) -> str:
    """Coordinate economic analysis across market and financial domains."""
    logger.info("📈 Economic Department coordinating analysis...")

    if isKorean(query):
        system_prompt = (
            "당신은 경제 부서 관리자입니다. 경제 분석을 조정하고 통합합니다."
            "시장 관련 질문에는 market_research 도구를 사용하세요."
            "경제적 질문에는 financial_analysis 도구를 사용하세요."
            "결과를 통합하여 통합된 경제 관점을 제공하세요."
            "중요: 질문이 명확하게 한 영역에 집중되지 않는 한 두 도구를 모두 사용하여 철저한 분석을 수행하세요."
        )
    else:
        system_prompt = (
            "You are an economic department manager who coordinates specialized economic analyses. "
            "For market-related questions, use the market_research tool. "
            "For financial questions, use the financial_analysis tool. "
            "Synthesize the results into a cohesive economic perspective. "
            "Important: Make sure to use both tools for comprehensive analysis unless the query is clearly focused on just one area."
        )

    econ_manager = Agent(
        system_prompt=system_prompt,
        tools=[market_research, financial_analysis],
        callback_handler=None
    )

    agent_stream = econ_manager.stream_async(query)
    result = await show_streams(agent_stream, containers)

    return result
```


## 설치하기

Repository를 clone 합니다.

```text
git clone https://github.com/kyopark2014/strands-agent/
```

필요한 라이브러리를 설치합니다. 여기서 필요한 라이브러리는 strands-agents, strands-agents-tools 입니다.

```python
cd strands-agent && pip install -r requirements.txt
```

CDK로 구동이 필요한 인프라인 CloudFront, S3, OpenSearch, Knowledge base, tavily, weather등의 secret을 설치합니다. 만약 cdk boootstraping이 안되어 있다면 설치후 수행합니다.

```text
cd cdk-strands-agent/ && cdk deploy --all
```

설치가 완료되면, 아래와 같이 "CdkStrandsAgentStack.environmentforstrandsagent"를 복사하여 application/config.json 파일을 생성합니다.

![image](https://github.com/user-attachments/assets/386edb27-ed29-49df-9df1-447b457e70ec)

config.json은 strands agent의 동작에 필요한 정보를 가지고 있고, [.gitignore](./application/.gitignore)에 의해 git으로 공유 되지 않습니다. 생성된 config.json의 셈플은 아래와 같습니다.

```json
{
   "projectName":"strands-agent",
   "accountId":"923476740234",
   "region":"us-west-2",
   "knowledge_base_role":"arn:aws:iam::923476740234:role/role-knowledge-base-for-strands-agent-us-west-2",
   "collectionArn":"arn:aws:aoss:us-west-2:923476740234:collection/lg29d83r30h8vrj4h4a",
   "opensearch_url":"https://lg29d83r30h8vrj4h4a.us-west-2.aoss.amazonaws.com",
   "s3_bucket":"storage-for-strands-agent-923476740234-us-west-2",
   "s3_arn":"arn:aws:s3:::storage-for-strands-agent-923476740234-us-west-2",
   "sharing_url":"https://a114n31pi9f63b.cloudfront.net"
}
```

이후 [Secret Manager](https://us-west-2.console.aws.amazon.com/secretsmanager/listsecrets?region=us-west-2)에 접속하여 아래와 같은 credential을 입력합니다.

![image](https://github.com/user-attachments/assets/a29ed9ab-86ff-4076-8ca7-7be8122a38a6)

만약 streamlit이 설치되어 있지 않다면 [streamlit](https://docs.streamlit.io/get-started/installation)을 참조하여 설치합니다. 이후 아래와 같이 실행합니다.

```text
streamlit run application/app.py
```

실행하면 아래와 같은 화면이 보여집니다. Agent를 선택하면 Strands agent를 실행하고 동작을 확인할 수 있습니다. 적절한 MCP 서버와 모델을 필요에 따라 선택하여 활용합니다. 

![image](https://github.com/user-attachments/assets/36337750-9321-452b-a59b-2fa611ef576d)


### 실행 결과

"us-west-2의 AWS bucket 리스트는?"와 같이 입력하면, aws cli를 통해 필요한 operation을 수행하고 얻어진 결과를 아래와 같이 보여줍니다.

<img src="https://github.com/user-attachments/assets/d7a99236-185b-4361-8cbf-e5a45de07319" width="600">


MCP로 wikipedia를 설정하고 "strand에 대해 설명해주세요."라고 질문하면 wikipedia의 search tool을 이용하여 아래와 같은 결과를 얻습니다.

<img src="https://github.com/user-attachments/assets/f46e7f47-65e0-49d8-a5c0-49e834ff5de8" width="600">


특정 Cloudwatch의 로그를 읽어서, 로그의 특이점을 확인할 수 있습니다.

<img src="https://github.com/user-attachments/assets/da48a443-bd53-4c2f-a083-cfcd4e954360" width="600">

"Image generation" MCP를 선택하고, "AWS의 한국인 solutions architect의 모습을 그려주세요."라고 입력하면 아래와 같이 이미지를 생성할 수 있습니다.

<img src="https://github.com/user-attachments/assets/a0b46a64-5cb7-4261-82df-b5d4095fdfd2" width="600">


## Reference

[Strands Agents SDK](https://strandsagents.com/0.1.x/)

[Strands Agents Samples](https://github.com/strands-agents/samples/tree/main)

[Example Built-in Tools](https://strandsagents.com/0.1.x/user-guide/concepts/tools/example-tools-package/)

[Introducing Strands Agents, an Open Source AI Agents SDK](https://aws.amazon.com/ko/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)

[use_aws.py](https://github.com/strands-agents/tools/blob/main/src/strands_tools/use_aws.py)

[Strands Agents와 오픈 소스 AI 에이전트 SDK 살펴보기](https://aws.amazon.com/ko/blogs/tech/introducing-strands-agents-an-open-source-ai-agents-sdk/)

[Drug Discovery Agent based on Amazon Bedrock](https://github.com/hsr87/drug-discovery-agent)

[Strands Agent - Swarm](https://strandsagents.com/latest/user-guide/concepts/multi-agent/swarm/)
