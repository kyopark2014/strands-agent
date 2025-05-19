import logging
import sys

from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, current_time

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    max_tokens=64000,
    # boto_client_config=Config(
    #    read_timeout=900,
    #    connect_timeout=900,
    #    retries=dict(max_attempts=3, mode="adaptive"),
    # ),
    additional_request_fields={
        "thinking": {
            "type": "disabled",
            # "budget_tokens": 2048,
        }
    },
)

system_prompt = """You are a helpful personal assistant that specializes in managing my appointments and calendar. 
You have access to appointment management tools, a calculator, and can check the current time to help me organize my schedule effectively. 
Always provide the appointment id so that I can update it if required"""

# Create an agent with default settings
agent = Agent(
    model=model,
    system_prompt=system_prompt,
    tools=[    
        calculator, 
        current_time    
    ],
)

# Ask the agent a question
question = "현재 시간은?"
result = agent(question)
logger.info(result)

# messages
logger.info(agent.messages)

# metrics
logger.info(result.metrics)

for m in agent.messages:
    for content in m["content"]:
        if "toolUse" in content:
            print("Tool Use:")
            tool_use = content["toolUse"]
            print("\tToolUseId: ", tool_use["toolUseId"])
            print("\tname: ", tool_use["name"])
            print("\tinput: ", tool_use["input"])
        if "toolResult" in content:
            print("Tool Result:")
            tool_result = m["content"][0]["toolResult"]
            print("\tToolUseId: ", tool_result["toolUseId"])
            print("\tStatus: ", tool_result["status"])
            print("\tContent: ", tool_result["content"])
            print("=======================")