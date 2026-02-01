from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])
response = agent("What is 80 / 4?")
