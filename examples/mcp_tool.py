from strands import Agent
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient
from strands_tools import calculator, current_time, use_aws

stdio_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"])
))

with stdio_mcp_client as client:
    aws_documentation_tools = client.list_tools_sync()

    tools=[calculator, current_time, use_aws]

    tools.extend(aws_documentation_tools)

    agent = Agent(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tools=tools
    )

    response = agent("AgentCore Runtime이란?")
    print(response)
