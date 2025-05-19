import utils
import info
import boto3
import traceback
import uuid
import json
import logging
import sys

from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, current_time, use_aws

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

model_name = "Claude 3.7 Sonnet"
model_type = "claude"
debug_mode = "Enable"
model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
models = info.get_model_info(model_name)
reasoning_mode = 'Disable'

def update(modelName, reasoningMode):    
    global model_name, model_id, model_type, debug_mode, multi_region
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]

#########################################################
# Strands Agent 
#########################################################
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


def run_strands_agent(question, history_mode, st):
    agent = create_agent()
    
    with st.status("thinking...", expanded=True, state="running") as status:    
        response = agent(question)
        logger.info(f"response: {response}")

        for msg in agent.messages:
            if "role" in msg:
                if msg["role"] == "assistant":
                    if "content" in msg:
                        content = msg["content"]

                        if isinstance(content, list):
                            if "text" in content[0]:
                                logger.info(f"Assistant: {content[0]['text']}")
                                st.info(f"{content[0]['text']}")

                            if "toolUse" in content[0]:
                                logger.info("Tool Use:")
                                tool_use = content[0]["toolUse"]
                                logger.info(f"\tToolUseId: {tool_use['toolUseId']}")
                                logger.info(f"\tname: {tool_use['name']}")
                                logger.info(f"\tinput: {tool_use['input']}")

                                st.info(f"{tool_use['name']}: {tool_use['input']}")

        # metrics
        logger.info(response.metrics)

        # all messages
        logger.info(f"messages: {agent.messages}")

        st.markdown(response)


        # print("Assistant: ", response.content)
        # print("Metrics: ", response.metrics)
        # print("Messages: ", response.messages)
        # print("Tool Use: ", response.messages[0]["content"][0]["toolUse"])
        # print("Tool Result: ", response.messages[0]["content"][0]["toolResult"])
        # print("=====================================")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

        return response