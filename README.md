# Strands Agent

Strands agent는 AI agent 구축 및 실행을 위해 설계된 오픈소스 SDK입니다. 계획(planning), 사고 연결(chaining thoughts), 도구 호출, Reflection과 같은 agent 기능을 쉽게 활용할 수 있습니다. 이를 통해 LLM model과 tool을 연결하며, 모델의 추론 능력을 이용하여 도구를 계획하고 실행합니다. Amazon Bedrock, Anthropic, Meta의 모델을 지원하며, Accenture, Anthropic, Meta와 같은 기업들이 참여하고 있습니다. 

[Strands Agents SDK](https://strandsagents.com/0.1.x/)

[Strands Agents Samples](https://github.com/strands-agents/samples/tree/main)

[Example Built-in Tools](https://strandsagents.com/0.1.x/user-guide/concepts/tools/example-tools-package/)


cache_prompt: Cache point type for the system prompt
cache_tools: Cache point type for tools
guardrail_id: ID of the guardrail to apply
guardrail_trace: Guardrail trace mode. Defaults to enabled.
guardrail_version: Version of the guardrail to apply
guardrail_stream_processing_mode: The guardrail processing mode
guardrail_redact_input: Flag to redact input if a guardrail is triggered. Defaults to True.
guardrail_redact_input_message: If a Bedrock Input guardrail triggers, replace the input with this message.
guardrail_redact_output: Flag to redact output if guardrail is triggered. Defaults to False.
guardrail_redact_output_message: If a Bedrock Output guardrail triggers, replace output with this message.
max_tokens: Maximum number of tokens to generate in the response
model_id: The Bedrock model ID (e.g., "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
stop_sequences: List of sequences that will stop generation when encountered
temperature: Controls randomness in generation (higher = more random)
top_p: Controls diversity via nucleus sampling (alternative to temperature)
