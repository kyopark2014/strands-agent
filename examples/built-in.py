from strands import Agent
from strands_tools import http_request

agent = Agent(tools=[http_request])
print(agent("서울 날씨는?"))
