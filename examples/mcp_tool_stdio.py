from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient

stdio_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", 
        args=["awslabs.aws-documentation-mcp-server@latest"]))
)

with stdio_mcp_client:
    tools = stdio_mcp_client.list_tools_sync()
    agent = Agent(tools=tools)
    agent("AgentCore Runtime이란?")
