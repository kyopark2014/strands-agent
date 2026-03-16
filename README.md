# Strands Agent

여기에서는 [Strands agent](https://strandsagents.com/0.1.x/)를 이용해 Agentic AI를 구현하는것을 설명합니다. Strands Agent는 AI agent 구축 및 실행을 위해 설계된 오픈소스 SDK입니다. 계획(planning), 사고 연결(chaining thoughts), 도구 호출, Reflection과 같은 agent 기능을 쉽게 활용할 수 있습니다. 이를 통해 LLM model과 tool을 연결하며, 모델의 추론 능력을 이용하여 도구를 계획하고 실행합니다. 현재 Amazon Bedrock, Anthropic, Meta의 모델을 지원하며, Accenture, Anthropic, Meta와 같은 기업들이 참여하고 있습니다. 

여기에서 사용하는 architecture는 아래와 같습니다. Agent의 기본동작 확인 및 구현을 위해 EC2에 docker 형태로 탑재되어 ALB와 CloudFront를 이용해 사용자가 streamlit으로 동작을 테스트 할 수 있습니다. Agent가 생성하는 그림이나 문서는 S3를 이용해 공유될 수 있으며, EC2에 내장된 MCP server/client를 이용해 인터넷검색(Tavily), RAG(knowledge base) AWS tools(use-aws), AWS Document를 이용할 수 있습니다.

<img width="900" alt="image" src="https://github.com/user-attachments/assets/69327c04-ea88-4647-bfce-4e2cae6beba0" />





Strands agent는 아래와 같은 [Agent Loop](https://strandsagents.com/0.1.x/user-guide/concepts/agents/agent-loop/)을 가지고 있으므로, 적절한 tool을 선택하여 실행하고, reasoning을 통해 반복적으로 필요한 동작을 수행합니다. 

![image](https://github.com/user-attachments/assets/6f641574-9d0b-4542-b87f-98d7c2715e09)

Tool들을 아래와 같이 병렬로 처리할 수 있습니다.

```python
agent = Agent(
    max_parallel_tools=4  
)
```

## Strands Agent 활용 방법

### Streamlit에서 agent의 실행

[app.py](./application/app.py)와 같이 사용자가 "RAG", "Agent"을 선택할 수 있습니다. "Agent"은 Strands agent를 이용하여 MCP로 필요시 tool들을 이용하여 RAG등을 활용할 수 있습니다. Streamlit의 UI를 위하여 user의 입력과 결과인 response을 [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)로 관리합니다. 

```python
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        containers = {
            "tools": st.empty(),
            "status": st.empty(),
            "notification": [st.empty() for _ in range(1000)],
            "key": st.empty()
        }
        if mode == 'Agent':
            response, image_urls = asyncio.run(chat.run_strands_agent(
                query=prompt, 
                strands_tools=selected_strands_tools, 
                mcp_servers=selected_mcp_servers, 
                containers=containers))
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": image_urls if image_urls else []
        })
```

### Agent의 실행

아래와 같이 system prompt, model, tool 정보를 가지고 agent를 생성합니다.

```python
def create_agent(system_prompt, tools):
    if system_prompt==None:
        system_prompt = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    model = get_model()    
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        conversation_manager=conversation_manager
    )
    return agent
```

[chat.py](./application/chat.py)와 같이 Agent를 실행하고 stream으로 결과를 받아서 보여줍니다. 이때, 아래와 같이 event에서 "data"만을 추출한 후에 full_response로 저장한 후에 markdown으로 표시합니다. 

```python
async def run_strands_agent(query, strands_tools, mcp_servers, containers):
    await strands_agent.initiate_agent(
        system_prompt=None, 
        strands_tools=strands_tools, 
        mcp_servers=mcp_servers
    )

    final_result = current = ""
    with strands_agent.mcp_manager.get_active_clients(mcp_servers) as _:
        agent_stream = strands_agent.agent.stream_async(query)
        
        async for event in agent_stream:
            text = ""            
            if "data" in event:
                text = event["data"]
                logger.info(f"[data] {text}")
                current += text

            elif "result" in event:
                final = event["result"]                
                message = final.message
                if message:
                    content = message.get("content", [])
                    result = content[0].get("text", "")
                    final_result = result
    return final_result
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

    agent = Agent(
        model=model,
        system_prompt=system,
        tools=tools,
        conversation_manager=conversation_manager
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
for mcp_tool in mcp_servers:
    with mcp_manager.get_active_clients([mcp_tool]) as _:
        client = mcp_manager.get_client(mcp_tool)
        mcp_servers_list = client.list_tools_sync()
        tools.extend(mcp_servers_list)

agent = create_agent(system_prompt, tools)
tool_list = get_tool_list(tools)
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

Supervisor는 전문 agent인 collaborator로 hand off를 수행함으로써 더 향상된 답변을 얻을 수 있습니다. 

<img width="803" height="724" alt="strands_supervisor" src="https://github.com/user-attachments/assets/0fde6fc8-3ebb-4f1b-a1a6-dc7985d00940" />


### Swarm

[Multi-Agent Systems and Swarm Intelligence](https://strandsagents.com/latest/user-guide/concepts/multi-agent/swarm/)와 같이 Agent들이 서로 협조하면서 복잡한 문제를 해결 할 수 있습니다. 

#### Mesh Swarm

[Mesh Swarm Architecture](https://strandsagents.com/latest/user-guide/concepts/multi-agent/swarm/#mesh-swarm-architecture)와 같이 여러 agent들간의 협업을 수행하 수 있습니다. Research agent는 논리적인 답변을, creative agent는 흥미로운 답변을 제공합니다. 이때, critical agent로 두 agent들의 개선점을 도출한 후에, summarizer agent로 최적의 답변을 구할 수 있습니다.

<img width="700" alt="swarm" src="https://github.com/user-attachments/assets/b2d400b5-87f2-4a1a-9e28-877e107834c2" />


이를 구현하는 방법은 [strands_swarm.py](./application/strands_swarm.py)을 참조합니다. 각 agent의 페르소나에 맞게 MCP tool과 함께 agent를 정의 합니다.

```python
# Create specialized agents with different expertise
# research agent
system_prompt = (
    "당신은 정보 수집과 분석을 전문으로 하는 연구원입니다. "
    "당신의 역할은 해당 주제에 대한 사실적 정보와 연구 통찰력을 제공하는 것입니다. "
    "정확한 데이터를 제공하고 문제의 핵심적인 측면들을 파악하는 데 집중해야 합니다. "
    "다른 에이전트로부터 입력을 받을 때, 그들의 정보가 당신의 연구와 일치하는지 평가하세요. "
)
model = strands_agent.get_model()
research_agent = Agent(
    model=model,
    system_prompt=system_prompt, 
    tools=tools
)

# Creative Agent
system_prompt = (
    "당신은 혁신적인 솔루션 생성을 전문으로 하는 창의적 에이전트입니다. "
    "당신의 역할은 틀에 박힌 사고에서 벗어나 창의적인 접근법을 제안하는 것입니다. "
    "다른 에이전트들로부터 얻은 정보를 바탕으로 하되, 당신만의 독창적인 창의적 관점을 추가해야 합니다. "
    "다른 사람들이 고려하지 않았을 수도 있는 참신한 접근법에 집중하세요. "
)
creative_agent = Agent(
    model=model,
    system_prompt=system_prompt, 
    tools=tools
)

# Critical Agent
system_prompt = (
    "당신은 제안서를 분석하고 결함을 찾는 것을 전문으로 하는 비판적 에이전트입니다. "
    "당신의 역할은 다른 에이전트들이 제안한 해결책을 평가하고 잠재적인 문제점들을 식별하는 것입니다. "
    "제안된 해결책을 신중히 검토하고, 약점이나 간과된 부분을 찾아내며, 개선 방안을 제시해야 합니다. "
    "비판할 때는 건설적으로 하되, 최종 해결책이 견고하도록 보장하세요. "
)
critical_agent = Agent(
    model=model,
    system_prompt=system_prompt, 
    tools=tools
)

# summarizer agent
system_prompt = (
    "당신은 정보 종합을 전문으로 하는 요약 에이전트입니다. "
    "당신의 역할은 모든 에이전트로부터 통찰력을 수집하고 응집력 있는 최종 해결책을 만드는 것입니다."
    "최고의 아이디어들을 결합하고 비판점들을 다루어 포괄적인 답변을 만들어야 합니다. "
    "원래 질문을 효과적으로 다루는 명확하고 실행 가능한 요약을 작성하는 데 집중하세요. "
)
summarizer_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    callback_handler=None)
```

주어진 질문에 대해 research, creative, critical agent의 응답을 구하고, 자신의 결과와 함게 다른 agent들의 결과를 전달합니다.

```python
result = research_agent.stream_async(question)
research_result = await show_streams(result, containers)

result = creative_agent.stream_async(question)
creative_result = await show_streams(result, containers)

result = critical_agent.stream_async(question)
critical_result = await show_streams(result, containers)

research_messages = []
creative_messages = []
critical_messages = []

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

research, creative, critical agent들은 병렬로 실행이 가능합니다. 따라서 아래와 같은 형태로도 구현할 수 있습니다.

```python
tasks = [
    _research_agent_worker(research_agent, question, request_id),
    _creative_agent_worker(creative_agent, question, request_id),
    _critical_agent_worker(critical_agent, question, request_id)
]
results = await asyncio.gather(*tasks)
research_result, creative_result, critical_result = results

summarizer_agent = create_summarizer_agent(question, tools)
summarizer_messages = []
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

#### Swarm Tool

[Creating Swarm of agents using Strands Agents](https://github.com/strands-agents/samples/blob/main/01-tutorials/02-multi-agent-systems/02-swarm-agent/swarm.ipynb)에서 strands agent에서 swarm을 사용할 수 있도록 tool을 제공하고 있습니다. 이때 agent에서 설정할 수 있는 협업 옵션은 아래와 같습니다.

- Collaborative: Agents build upon others' insights and seek consensus
- Competitive: Agents develop independent solutions and unique perspectives
- Hybrid: Balances cooperation with independent exploration

협업하는 swarm agent들로부터 얻어진 결과를 summarized agent로 정리하여 답변합니다. 아래는 swarm tool을 사용할때의 diagram입니다. 여기서 swarm agent의 숫자는 swarm_size로 조정합니다.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/27129887-a62f-403f-abb3-2d650fcbcaa1" />

[strands_swarm_tool.py](./application/strands_swarm_tool.py)와 같이 strands agent를 이용해 swarm 형태의 multi agent를 구현하고, 이를 통해 복잡한 문제를 풀 수 있습니다.

```python
from strands_tools import swarm

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

이때의 결과는 아래와 같습니다. 전문 agent에 대한 role과 prompt를 생성한 후에 요약된 결과를 보여줍니다.


<img width="723" height="789" alt="strands_swarm_tool" src="https://github.com/user-attachments/assets/f0b43cfb-abda-4c57-b1f6-c553f988097f" />


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

### Graph with Loops: Plan and Execute

[strands_plan_and_execute.py](./application/strands_plan_and_execute.py)에서는 plan and execute pattern의 agent를 구현합니다. "planner"에서 먼저 plan을 생성한 후에 executer가 결과를 구합니다. 이때, 모든 plan이 실행이 안되었다면 replanner가 새로운 계획을 세웁니다. 만약 모든 plan이 실행이 되었다면 synthesizer로 전환되어 최종 결과를 얻습니다. 

<img width="400" alt="image" src="https://github.com/user-attachments/assets/5f9462c9-c3d5-4bd2-a1e0-69334a69a70e" />

상세한 코드는 [strands_plan_and_execute.py](./application/strands_plan_and_execute.py)를 참조합니다.

```python
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
```

### Graph with Loops: Multi-Agent Feedback Cycles

[Graph with Loops - Multi-Agent Feedback Cycles](https://strandsagents.com/1.x/documentation/docs/examples/python/graph_loops_example/)을 이용해 아래와 같은 feedback loop을 구현합니다.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/3346072b-510a-42a5-8d6d-07250683de72" />

상세한 코드는 [strands_graph_with_loop.py](./application/strands_graph_with_loop.py)을 참조합니다. 이 코드는 [graph_loops_example.py](https://github.com/strands-agents/docs/blob/main/docs/examples/python/graph_loops_example.py)을 참조하였습니다.

```python
checker = QualityChecker(approval_after=2)

builder = GraphBuilder()
builder.add_node(writer, "writer")
builder.add_node(checker, "checker") 
builder.add_node(finalizer, "finalizer")

builder.add_edge("writer", "checker")
builder.add_edge("checker", "writer", condition=needs_revision)
builder.add_edge("checker", "finalizer", condition=is_approved)
builder.set_entry_point("writer")

graph = builder.build()

result = await graph.invoke_async(question)
```




## Memory 활용하기

Chatbot은 연속적인 사용자의 상호작용을 통해 사용자의 경험을 향상시킬수 있습니다. 이를 위해 이전 대화의 내용을 새로운 대화에서 활용할 수 있어야하며, 일반적으로 chatbot은 sliding window를 이용해 새로운 transaction마다 이전 대화내용을 context로 제공해야 했습니다. 여기에서는 필요한 경우에만 이전 대화내용을 참조할 수 있도록 short term/long term 메모리를 MCP를 이용해 활용합니다. 이렇게 하면 context에 불필요한 이전 대화가 포함되지 않아서 사용자의 의도를 명확히 반영하고 비용도 최적화 할 수 있습니다. 

### Short Term Memory

Short term memory를 위해서는 대화 transaction을 아래와 같이 agentcore의 memory에 저장합니다. 상세한 코드는 [agentcore_memory.py](./application/agentcore_memory.py)을 참조합니다.

```python
def save_conversation_to_memory(memory_id, actor_id, session_id, query, result):
    event_timestamp = datetime.now(timezone.utc)
    conversation = [
        (query, "USER"),
        (result, "ASSISTANT")
    ]
    memory_result = memory_client.create_event(
        memory_id=memory_id,
        actor_id=actor_id, 
        session_id=session_id, 
        event_timestamp=event_timestamp,
        messages=conversation
    )
```

이후, 대화중에 사용자의 이전 대화정보가 필요하다면, [mcp_server_short_term_memory.py](./application/mcp_server_short_term_memory.py)와 같이 memory, actor, session로 max_results 만큼의 이전 대화를 조회하여 활용합니다.  

```python
events = client.list_events(
    memory_id=memory_id,
    actor_id=actor_id,
    session_id=session_id,
    max_results=max_results
)
```

### Long Term Memory

Long term meory를 위해 필요한 정보에는 memory, actor, session, namespace가 있습니다. 아래와 같이 이미 저장된 값이 있다면 가져오고, 없다면 생성합니다. 상세한 코드는 [strands_agent.py](./application/strands_agent.py)을 참조합니다.

```python
# initate memory variables
memory_id, actor_id, session_id, namespace = agentcore_memory.load_memory_variables(chat.user_id)
logger.info(f"memory_id: {memory_id}, actor_id: {actor_id}, session_id: {session_id}, namespace: {namespace}")

if memory_id is None:
    # retrieve memory id
    memory_id = agentcore_memory.retrieve_memory_id()
    logger.info(f"memory_id: {memory_id}")        
    
    # create memory if not exists
    if memory_id is None:
        memory_id = agentcore_memory.create_memory(namespace)
    
    # create strategy if not exists
    agentcore_memory.create_strategy_if_not_exists(memory_id=memory_id, namespace=namespace, strategy_name=chat.user_id)

    # save memory variables
    agentcore_memory.update_memory_variables(
        user_id=chat.user_id, 
        memory_id=memory_id, 
        actor_id=actor_id, 
        session_id=session_id, 
        namespace=namespace)
```

생성형 AI 애플리케이션에서는 대화중 필요한 메모리 정보가 있다면 이를 MCP를 이용해 조회합니다. [mcp_server_long_term_memory.py](./application/mcp_server_long_term_memory.py)에서는 long term memory를 이용해 대화 이벤트를 저장하거나 조회할 수 있습니다. 아래는 신규로 레코드를 생성하는 방법입니다.

```python
response = create_event(
    memory_id=memory_id,
    actor_id=actor_id,
    session_id=session_id,
    content=content,
    event_timestamp=datetime.now(timezone.utc),
)
event_data = response.get("event", {}) if isinstance(response, dict) else {}
```

대화에 필요한 정보는 아래와 같이 조회합니다.

```python
contents = []
response = retrieve_memory_records(
    memory_id=memory_id,
    namespace=namespace,
    search_query=query,
    max_results=max_results,
    next_token=next_token,
)
relevant_data = {}
if isinstance(response, dict):
    if "memoryRecordSummaries" in response:
        relevant_data["memoryRecordSummaries"] = response["memoryRecordSummaries"]    
    for memory_record_summary in relevant_data["memoryRecordSummaries"]:
        json_content = memory_record_summary["content"]["text"]
        content = json.loads(json_content)
        contents.append(content)
```

아래와 같이 "내가 좋아하는 스포츠는?"를 입력하면 long term memory에서 사용자에 대한 정보를 조회하여 답변할 수 있습니다.

<img width="721" height="770" alt="image" src="https://github.com/user-attachments/assets/193105da-09df-4e28-bc64-b72a79936550" />


### MCP

Plugin의 Connector는 MCP를 이용해 구현합니다. 이때 필요한 MCP 설정은 아래를 참조합니다. 

- [Slack](https://github.com/kyopark2014/mcp/blob/main/mcp-slack.md): Slack 내용을 조회하고 메시지를 보낼 수 있습니다. SLACK_TEAM_ID, SLACK_BOT_TOKEN으로 설정합니다.

- [Tavily](https://github.com/kyopark2014/mcp/blob/main/mcp-tavily.md): Tavily를 이용해 인터넷을 검색합니다. [installer.py](./installer.py)에서 secret으로 설정후에 [utils.py](./application/utils.py)에서 TAVILY_API_KEY로 등록하여 활용합니다.

- [RAG](https://github.com/kyopark2014/mcp/blob/main/mcp-rag.md): Knowledge Base를 이용해 RAG를 활용합니다. IAM 인증을 이용하므로 별도로 credential 설정하지 않습니다.

- [web_fetch](https://github.com/kyopark2014/mcp/blob/main/mcp-web-fetch.md): playwright기반으로 url의 문서를 markdown으로 불러올 수 있습니다. 별도 인증이 필요하지 않습니다.

- [Google 메일/캘린더](https://github.com/kyopark2014/mcp/blob/main/mcp-gog.md): 구글 메일을 조회하거나 보낼 수 있습니다. Gog CLI를 설치하여 google 인증을 통해 활용합니다.

- [Notion](https://github.com/kyopark2014/mcp/blob/main/mcp-notion.md): Notion을 읽거나 쓸 수 있습니다. [installer.py](./installer.py)에서 secret으로 설정후에 [utils.py](./application/utils.py)에서 NOTION_TOKEN을 등록하여 활용합니다.

- [text_extraction](https://github.com/kyopark2014/mcp/blob/main/mcp-text-extraction.md): 이미지의 텍스트를 추출합니다. 별도 인증이 필요하지 않습니다.


### Plugin

[Plugin 폴더](./application/plugins)에 plugin을 저장하면 [plugin.py](./application/plugin.py)와 같이 pluin 리스트를 가져와서 왼쪽 메뉴에 표시합니다.

```python
def available_plugins_list():
    plugin_dir = PLUGINS_DIR
    if not os.path.isdir(plugin_dir):
        return []
    
    plugin_list = []
    for plugin in os.listdir(plugin_dir):
        plugin_list.append({"name": plugin})
        
    return plugin_list
```

사용자가 plugin을 선택하면 아래와 같이 [plugin_agent.py](./application/plugin_agent.py)의 run_plugin_agent을 실행합니다. 
  
```python
for plugin in plugin_list:
    if mode == plugin["name"]:
        with st.status("thinking...", expanded=True, state="running") as status:
            containers = {
                "tools": st.empty(),
                "status": st.empty(),
                "notification": [st.empty() for _ in range(1000)],
                "key": st.empty()
            }
        response, image_urls = asyncio.run(plugin_agent.run_plugin_agent(prompt, selected_strands_tools, selected_mcp_servers, plugin["name"], containers))
```

이때, [strands_agent.py](./application/strands_agent.py)와 같이 mcp_server를 초기화하고 strands와 MCP로 제공되는 tool을 설정합니다. 이후에 아래의 get_builtin_tools으로 SKILL 동작에 필요한 tool을 등록합니다. 이때, Strands나 MCP tool들이 SKILL에 필요한 tool들과 같은 이름이 있는지 확인합니다.

```python
init_mcp_clients(mcp_servers)
tools = update_tools(strands_tools, mcp_servers)

builtin_tools = get_builtin_tools()

tool_names = {tool.tool_name for tool in tools}
for bt in builtin_tools:
    if bt.tool_name not in tool_names:
        tools.append(bt)
```

SKILL을 위해 파일을 읽고 쓰는 read_file, write_file 뿐 아니라 script등의 code를 실행하는 execute_code가 필요합니다. get_skill_instructions은 skil의 description을 가져올 때 필요한 tool입니다.

```python
def get_builtin_tools():
    """Return the list of built-in tools for the skill-aware agent."""
    return [execute_code, write_file, read_file, upload_file_to_s3, get_skill_instructions]
```

Agent를 생성할 때에 아래와 같이 plugin 이름과 command에 대한 정보가 필요합니다. 이때 build_system_prompt으로 SKILL/Command를 처리할 수 있는 prompt를 생성합니다.

```python
agent = create_agent(system_prompt, tools, plugin_name, command)

def create_agent(system_prompt: Optional[str], tools: list, plugin_name: Optional[str], command: Optional[str] = None):
    if system_prompt==None:
        system_prompt = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )

    if not system_prompt or not system_prompt.strip():
        system_prompt = "You are a helpful AI assistant."
    
    # add skills metadata to system prompt
    system_prompt = build_system_prompt(system_prompt, plugin_name, command)

    model = get_model()    
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        conversation_manager=conversation_manager,
    )

    return agent
```

SKILL은 아래와 같이 SKILL_SYSTEM_PROMPT와 실행에 필요한 경로를 가지고 있습니다.

```python
SKILL_USAGE_GUIDE = (
    "\n## Skill 사용 가이드\n"
    "위의 <available_skills>에 나열된 skill이 사용자의 요청과 관련될 때:\n"
    "1. 먼저 get_skill_instructions 도구로 해당 skill의 상세 지침을 로드하세요.\n"
    "2. 지침에 포함된 코드 패턴을 execute_code 도구로 실행하세요.\n"
    "3. skill 지침이 없는 일반 질문은 직접 답변하세요.\n"
)

def build_skill_prompt(plugin_name: str) -> str:
    """Build skill-related prompt: path info, available skills XML, and usage guide."""
    skill_info = selected_skill_info(plugin_name)
    logger.info(f"plugin_name: {plugin_name}, skill_info: {skill_info}")

    if plugin_name != "base":
        default_skill_info = selected_skill_info("base")
        if default_skill_info:
            skill_info.extend(default_skill_info)
            logger.info(f"default_skill_info: {default_skill_info}")

    path_info = (
        f"## Paths (use absolute paths for write_file, read_file)\n"
        f"- WORKING_DIR: {WORKING_DIR}\n"
        f"- ARTIFACTS_DIR: {ARTIFACTS_DIR}\n"
        f"Example: write_file(filepath='{os.path.join(ARTIFACTS_DIR, 'report.drawio')}', content='...')\n\n"
    )

    skills_xml = get_skills_xml(skill_info)
    if skills_xml:
        return f"{SKILL_SYSTEM_PROMPT}\n{path_info}\n{skills_xml}\n{SKILL_USAGE_GUIDE}"
    return f"{SKILL_SYSTEM_PROMPT}\n{path_info}"
```

Command는 사용자의 입력이 슬랙시(/)로 시작할 때에 첫번째 단어를 추출해 활용합니다. 이때, is_command로 해당 plugin에 command 이름을 가지는 markdown 파일이 있는지 확인합니다.

```python
command = None
if plugin.is_command(query, plugin_name):
    command = query.split(" ")[0].lstrip("/")
```

Command를 위한 prompt는 아래와 같습니다. Command의 경우에 SKILL_SYSTEM_PROMPT, 경로, skill 리스트, COMMAND_USAGE_GUIDE를 포함합니다.

```python
COMMAND_USAGE_GUIDE = (
    "\n## Command 사용 가이드\n"
    "위의 <command_instructions>에 따라 사용자 요청을 처리하세요.\n"
    "필요한 경우 get_skill_instructions로 skill 지침을 추가 로드하거나, execute_code, write_file 등 도구를 사용하세요.\n"
)

def build_command_prompt(plugin_name: str, command: str) -> str:
    """Build prompt for command mode: path info, command instructions, and available skills."""
    skill_info = selected_skill_info(plugin_name)
    logger.info(f"plugin_name: {plugin_name}, command: {command}, skill_info: {skill_info}")

    if plugin_name != "base":
        default_skill_info = selected_skill_info("base")
        if default_skill_info:
            skill_info.extend(default_skill_info)
            logger.info(f"default_skill_info: {default_skill_info}")

    path_info = (
        f"## Paths (use absolute paths for write_file, read_file)\n"
        f"- WORKING_DIR: {WORKING_DIR}\n"
        f"- ARTIFACTS_DIR: {ARTIFACTS_DIR}\n"
        f"Example: write_file(filepath='{os.path.join(ARTIFACTS_DIR, 'report.drawio')}', content='...')\n\n"
    )

    command_instructions = get_command_instructions(plugin_name, command)
    command_section = f"## Command Instructions\n<command_instructions>\n{command_instructions}\n</command_instructions>\n\n"

    skills_xml = get_skills_xml(skill_info)
    skills_section = f"{skills_xml}\n" if skills_xml else ""

    return f"{SKILL_SYSTEM_PROMPT}\n{path_info}\n{command_section}\n{skills_section}\n{COMMAND_USAGE_GUIDE}"
```



## 배포하기

### EC2로 배포하기

AWS console의 EC2로 접속하여 [Launch an instance](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)를 선택합니다. [Launch instance]를 선택한 후에 적당한 Name을 입력합니다. (예: es) key pair은 "Proceed without key pair"을 선택하고 넘어갑니다. 

<img width="700" alt="ec2이름입력" src="https://github.com/user-attachments/assets/c551f4f3-186d-4256-8a7e-55b1a0a71a01" />


Instance가 준비되면 [Connet] - [EC2 Instance Connect]를 선택하여 아래처럼 접속합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/e8a72859-4ac7-46af-b7ae-8546ea19e7a6" />

이후 아래와 같이 python, pip, git, boto3를 설치합니다.

```text
sudo yum install python3 python3-pip git docker -y
pip install boto3
```

Workshop의 경우에 아래 형태로 된 Credential을 복사하여 EC2 터미널에 입력합니다.

<img width="700" alt="credential" src="https://github.com/user-attachments/assets/261a24c4-8a02-46cb-892a-02fb4eec4551" />

아래와 같이 git source를 가져옵니다.

```python
git clone https://github.com/kyopark2014/strands-agent
```

아래와 같이 installer.py를 이용해 설치를 시작합니다.

```python
cd strands-agent && python3 installer.py
```

API 구현에 필요한 credential은 secret으로 관리합니다. 따라서 설치시 필요한 credential 입력이 필요한데 아래와 같은 방식을 활용하여 미리 credential을 준비합니다. 

- 일반 인터넷 검색: [Tavily Search](https://app.tavily.com/sign-in)에 접속하여 가입 후 API Key를 발급합니다. 이것은 tvly-로 시작합니다.  
- 날씨 검색: [openweathermap](https://home.openweathermap.org/api_keys)에 접속하여 API Key를 발급합니다. 이때 price plan은 "Free"를 선택합니다.

설치가 완료되면 아래와 같은 CloudFront로 접속하여 동작을 확인합니다. 

<img width="500" alt="cloudfront_address" src="https://github.com/user-attachments/assets/7ab1a699-eefb-4b55-b214-23cbeeeb7249" />


인프라가 더이상 필요없을 때에는 uninstaller.py를 이용해 제거합니다.

```text
python uninstaller.py
```


### 배포된 Application 업데이트 하기

AWS console의 EC2로 접속하여 [Launch an instance](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)를 선택하여 아래와 같이 아래와 같이 "app-for-es-us"라는 이름을 가지는 instance id를 선택합니다.

<img width="750" alt="image" src="https://github.com/user-attachments/assets/7d6d756a-03ba-4422-9413-9e4b6d3bc1da" />

[connect]를 선택한 후에 Session Manager를 선택하여 접속합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d1119cd6-08fb-4d3e-b1c2-77f2d7c1216a" />

이후 아래와 같이 업데이트한 후에 다시 브라우저에서 확인합니다.

```text
cd ~/strands-agent/ && sudo ./update.sh
```

### 실행 로그 확인

[EC2 console](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)에서 "app-for-es-us"라는 이름을 가지는 instance id를 선택 한 후에, EC2의 Session Manager를 이용해 접속합니다. 

먼저 아래와 같이 현재 docker container ID를 확인합니다.

```text
sudo docker ps
```

이후 아래와 같이 container ID를 이용해 로그를 확인합니다.

```text
sudo docker logs [container ID]
```

실제 실행시 결과는 아래와 같습니다.

<img width="600" src="https://github.com/user-attachments/assets/2ca72116-0077-48a0-94be-3ab15334e4dd" />

### Local에서 실행하기

AWS 환경을 잘 활용하기 위해서는 [AWS CLI를 설치](https://docs.aws.amazon.com/ko_kr/cli/v1/userguide/cli-chap-install.html)하여야 합니다. EC2에서 배포하는 경우에는 별도로 설치가 필요하지 않습니다. Local에 설치시는 아래 명령어를 참조합니다.

```text
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" 
unzip awscliv2.zip
sudo ./aws/install
```

AWS credential을 아래와 같이 AWS CLI를 이용해 등록합니다.

```text
aws configure
```

설치하다가 발생하는 각종 문제는 [Kiro-cli](https://aws.amazon.com/ko/blogs/korea/kiro-general-availability/)를 이용해 빠르게 수정합니다. 아래와 같이 설치할 수 있지만, Windows에서는 [Kiro 설치](https://kiro.dev/downloads/)에서 다운로드 설치합니다. 실행시는 셀에서 "kiro-cli"라고 입력합니다. 

```python
curl -fsSL https://cli.kiro.dev/install | bash
```

venv로 환경을 구성하면 편리하게 패키지를 관리합니다. 아래와 같이 환경을 설정합니다.

```text
python -m venv .venv
source .venv/bin/activate
```

이후 다운로드 받은 github 폴더로 이동한 후에 아래와 같이 필요한 패키지를 추가로 설치 합니다.

```text
pip install -r requirements.txt
```

이후 아래와 같은 명령어로 streamlit을 실행합니다. 

```text
streamlit run application/app.py
```



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

[Strands Python Example](https://github.com/strands-agents/docs/tree/main/docs/examples/python)

[Strands Agents SDK](https://strandsagents.com/0.1.x/)

[Strands Agents Samples](https://github.com/strands-agents/samples/tree/main)

[Example Built-in Tools](https://strandsagents.com/0.1.x/user-guide/concepts/tools/example-tools-package/)

[Introducing Strands Agents, an Open Source AI Agents SDK](https://aws.amazon.com/ko/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)

[use_aws.py](https://github.com/strands-agents/tools/blob/main/src/strands_tools/use_aws.py)

[Strands Agents와 오픈 소스 AI 에이전트 SDK 살펴보기](https://aws.amazon.com/ko/blogs/tech/introducing-strands-agents-an-open-source-ai-agents-sdk/)

[Drug Discovery Agent based on Amazon Bedrock](https://github.com/hsr87/drug-discovery-agent)

[Strands Agent - Swarm](https://strandsagents.com/latest/user-guide/concepts/multi-agent/swarm/)

[Strands Agent Streamlit Demo](https://github.com/NB3025/strands-streamlit-chat-demo)


[생성형 AI로 AWS 보안 점검 자동화하기: Q CLI에서 Strands Agents까지](https://catalog.us-east-1.prod.workshops.aws/workshops/89fc3def-0260-4fa7-91ce-623ad9a4d04a/ko-KR)

[AI Agent를 활용한 EKS 애플리케이션 및 인프라 트러블슈팅](https://catalog.us-east-1.prod.workshops.aws/workshops/bbd8a1df-c737-4f88-9d19-17bcecb7e712/ko-KR)

[Strands Agents 및 AgentCore와 함께하는 바이오·제약 연구 어시스턴트 구현하기](https://catalog.us-east-1.prod.workshops.aws/workshops/fe97ac91-ff75-4753-a269-af39e7c3d765/ko-KR)

[Strands Agents & Amazon Bedrock AgentCore 워크샵](https://github.com/hsr87/strands-agents-for-life-science)

[Agentic AI로 구현하는 리뷰 관리 자동화](https://catalog.us-east-1.prod.workshops.aws/workshops/59ea75b5-532c-4b57-982e-e58152ae5c46/ko-KR)

[Strands Agent Workshop (한국어)](https://github.com/chloe-kwak/strands-agent-workshop)

[Agentic AI Workshop: AI Fund Manager](https://catalog.us-east-1.prod.workshops.aws/workshops/a8702b51-fcf3-43b3-8d37-511ef1b38688/ko-KR)

[Agentic AI 펀드 매니저](https://github.com/ksgsslee/investment_advisor_strands)

[Workshop - Strands SDK와 AgentCore를 활용한 에이전틱 AI](https://catalog.workshops.aws/strands/ko-KR)
