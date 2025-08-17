#Initialize an agent with agent_graph capability
from strands.multiagent import GraphBuilder
from strands import Agent

mesh_agent = Agent()

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

print("============================================================")
print("============================================================")

#Execute task on newly built graph
result = graph("우리 회사는 새로운 AI 기반 고객 서비스 플랫폼 출시를 고려하고 있습니다. 초기 투자금은 200만 달러이며 3년간 예상 투자수익률(ROI)은 150%입니다. 재무적 평가는 어떻게 하시겠습니까?")
print("\n")
print("============================================================")
print("============================================================")

print(f"Response: {result}")

print("=============Node execution order:==========================")
print("============================================================")

# See which nodes were executed and in what order
for node in result.execution_order:
    print(f"Executed: {node.node_id}")

print("=============Graph metrics:=================================")
print("============================================================")


# Get performance metrics
print(f"Total nodes: {result.total_nodes}")
print(f"Completed nodes: {result.completed_nodes}")
print(f"Failed nodes: {result.failed_nodes}")
print(f"Execution time: {result.execution_time}ms")
print(f"Token usage: {result.accumulated_usage}")


# Get results from specific nodes
print("Financial Advisor:")
print("============================================================")
print("============================================================")
print(result.results["finance_expert"].result)
print("\n")

print("Technical Expert:")
print("============================================================")
print("============================================================")
print(result.results["tech_expert"].result)
print("\n")

print("Market Researcher:")
print("============================================================")
print("============================================================")
print(result.results["market_expert"].result)
print("\n")    