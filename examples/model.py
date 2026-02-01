from strands import Agent
from strands.models import BedrockModel

bedrock_model = BedrockModel(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

agent = Agent(model=bedrock_model)
response = agent("Bedrock이 뭐야?")
