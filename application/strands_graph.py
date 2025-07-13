# Workflow agents
# motivated from: https://strandsagents.com/latest/user-guide/concepts/multi-agent/workflow/terms/workflow-agent/

import chat
import os
import logging
import sys
import re
import strands_agent
import json

from strands import Agent, tool

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("strands-agent")

index = 0
def add_notification(containers, message):
    global index
    containers['notification'][index].info(message)
    index += 1

def add_response(containers, message):
    global index
    containers['notification'][index].markdown(message)
    index += 1

status_msg = []
def get_status_msg(status):
    global status_msg
    status_msg.append(status)

    if status != "end)":
        status = " -> ".join(status_msg)
        return "[status]\n" + status + "..."
    else: 
        status = " -> ".join(status_msg)
        return "[status]\n" + status    

os.environ["BYPASS_TOOL_CONSENT"] = "true"

async def show_streams(agent_stream, containers):
    tool_name = ""
    result = ""
    current_response = ""

    async for event in agent_stream:
        # logger.info(f"event: {event}")
        if "message" in event:
            message = event["message"]
            logger.info(f"message: {message}")

            for content in message["content"]:                
                if "text" in content:
                    logger.info(f"text: {content['text']}")
                    if chat.debug_mode == 'Enable':
                        add_response(containers, content['text'])

                    result = content['text']
                    current_response = ""

                if "toolUse" in content:
                    tool_use = content["toolUse"]
                    logger.info(f"tool_use: {tool_use}")
                    
                    tool_name = tool_use["name"]
                    input = tool_use["input"]
                    
                    logger.info(f"tool_nmae: {tool_name}, arg: {input}")
                    if chat.debug_mode == 'Enable':       
                        add_notification(containers, f"tool name: {tool_name}, arg: {input}")
                        containers['status'].info(get_status_msg(f"{tool_name}"))
            
                if "toolResult" in content:
                    tool_result = content["toolResult"]
                    logger.info(f"tool_name: {tool_name}")
                    logger.info(f"tool_result: {tool_result}")
                    logger.info(f"tool_result status: {tool_result.get('status', 'unknown')}")
                    logger.info(f"tool_result toolUseId: {tool_result.get('toolUseId', 'unknown')}")
                    
                    if "content" in tool_result:
                        tool_content = tool_result['content']
                        logger.info(f"tool_content length: {len(tool_content)}")
                        for i, content in enumerate(tool_content):
                            logger.info(f"content[{i}]: {content}")
                            if "text" in content:
                                text_value = content['text']
                                logger.info(f"text_value: {text_value}")
                                logger.info(f"text_value type: {type(text_value)}")
                                
                                # 코루틴 객체 문자열인지 확인
                                if isinstance(text_value, str) and '<coroutine object' in text_value:
                                    logger.info("Detected coroutine string, tool may still be executing...")
                                    if chat.debug_mode == 'Enable':
                                        add_notification(containers, f"tool result: Tool execution in progress...")
                                else:
                                    if chat.debug_mode == 'Enable':
                                        add_notification(containers, f"tool result: {text_value}")

        if "data" in event:
            text_data = event["data"]
            current_response += text_data

            if chat.debug_mode == 'Enable':
                containers["notification"][index].markdown(current_response)
            continue
    
    logger.info(f"show_streams completed, final result: {result}")
    logger.info(f"show_streams result type: {type(result)}")
    return result

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False

async def run_graph(question, containers):
    global status_msg
    status_msg = []

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"(start"))    

    # Level 3 - Specialized Analysis Agents
    @tool
    async def market_research(query: str) -> str:
        """Analyze market trends and consumer behavior."""
        logger.info("🔍 Market Research Specialist analyzing...")
        market_agent = Agent(
            system_prompt="You are a market research specialist who analyzes consumer trends, market segments, and purchasing behaviors. Provide detailed insights on market conditions, consumer preferences, and emerging trends.",
            callback_handler=None
        )

        add_notification(containers, f"market_research: {query}")
        agent_stream = market_agent.stream_async(query)
        result = await show_streams(agent_stream, containers)
        logger.info(f"market_research completed, result: {result}")
        logger.info(f"market_research result type: {type(result)}")
        return result

    @tool
    async def financial_analysis(query: str) -> str:
        """Analyze financial aspects and economic implications."""
        logger.info("💹 Financial Analyst processing...")
        if isKorean(query):
            system_prompt = (
                "당신은 경제 예측, 비용 효과 분석, 그리고 재무 모델링에 특화된 경제 분석가입니다."
                "재무적 가치, 경제적 영향, 그리고 예산 고려사항에 대한 통찰을 제공하세요."
            )
        else:
            system_prompt = (
                "You are a financial analyst who specializes in economic forecasting, cost-benefit analysis, and financial modeling. "
                 "Provide insights on financial viability, economic impacts, and budgetary considerations.",
            )
        financial_agent = Agent(
            system_prompt=system_prompt,
            callback_handler=None
        )

        add_notification(containers, f"financial_analysis: {query}")
        agent_stream = financial_agent.stream_async(query)
        result = await show_streams(agent_stream, containers)
        logger.info(f"financial_analysis completed, result: {result}")
        logger.info(f"financial_analysis result type: {type(result)}")

        return result

    @tool
    async def technical_analysis(query: str) -> str:
        """Analyze technical feasibility and implementation challenges."""
        logger.info("⚙️ Technical Analyst evaluating...")
        if isKorean(query):
            system_prompt = (
                "당신은 기술적 가능성, 구현 챌린지, 그리고 새로운 기술에 대한 평가를 하는 기술 분석가입니다."
                "기술적 측면, 구현 요구사항, 그리고 잠재적 기술적 장애에 대한 자세한 평가를 제공하세요."
            )
        else:
            system_prompt = (
                "You are a technology analyst who evaluates technical feasibility, implementation challenges, and emerging technologies. "
                "Provide detailed assessments of technical aspects, implementation requirements, and potential technological hurdles."
            )

        tech_agent = Agent(
            system_prompt=system_prompt,
            callback_handler=None
        )

        add_notification(containers, f"technical_analysis: {query}")
        agent_stream = tech_agent.stream_async(query)
        result = await show_streams(agent_stream, containers)
        logger.info(f"technical_analysis completed, result: {result}")
        logger.info(f"technical_analysis result type: {type(result)}")

        return result

    @tool
    async def social_analysis(query: str) -> str:
        """Analyze social impacts and behavioral implications."""
        logger.info("👥 Social Impact Analyst investigating...")

        if isKorean(query):
            system_prompt = (
                "당신은 변화가 지역사회, 행동, 그리고 사회 구조에 미치는 영향에 초점을 맞춘 사회적 영향 분석가입니다."
                "사회적인 의미, 예측되는 행동 변화, 그리고 지역사회 영향에 대한 통찰을 제공하세요."
            )            
        else:
            system_prompt = (
                "You are a social impact analyst who focuses on how changes affect communities, behaviors, and social structures. "
                "Provide insights on social implications, behavioral changes, and community impacts."
            )            
        
        social_agent = Agent(
            system_prompt=system_prompt,
            callback_handler=None
        )

        add_notification(containers, f"social_analysis: {query}")
        agent_stream = social_agent.stream_async(query)
        result = await show_streams(agent_stream, containers)
        logger.info(f"social_analysis completed, result: {result}")
        logger.info(f"social_analysis result type: {type(result)}")

        return result

    # Level 2 - Mid-level Manager Agent with its own specialized tools
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

        add_notification(containers, f"economic_department: {query}")
        agent_stream = econ_manager.stream_async(query)
        result = await show_streams(agent_stream, containers)
        logger.info(f"economic_department completed, result: {result}")
        logger.info(f"economic_department result type: {type(result)}")

        return result

    if isKorean(question):
        COORDINATOR_SYSTEM_PROMPT = (
            "당신은 복잡한 분석을 총괄하는 최고 경영자입니다."
            "경제적 질문에는 economic_department 도구를 사용하세요."
            "기술적 질문에는 technical_analysis 도구를 사용하세요."
            "사회적 질문에는 social_analysis 도구를 사용하세요."
            "결과를 통합하여 통합된 경제 관점을 제공하세요."
        )
    else:
        COORDINATOR_SYSTEM_PROMPT = (
            "You are an executive coordinator who oversees complex analyses across multiple domains."        
            "For economic questions, use the economic_department tool. "
            "For technical questions, use the technical_analysis tool. "
            "For social impact questions, use the social_analysis tool. "
            "Synthesize all analyses into comprehensive executive summaries. "
            "Your process should be: "
            "1. Determine which domains are relevant to the query (economic, technical, social) "
            "2. Collect analysis from each relevant domain using the appropriate tools "
            "3. Synthesize the information into a cohesive executive summary "
            "4. Present findings with clear structure and organization "
            "Always consider multiple perspectives and provide balanced, well-rounded assessments. "
        )

    # Create the coordinator agent with all tools
    coordinator = Agent(
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        tools=[economic_department, technical_analysis, social_analysis],
        callback_handler=None
    )

    add_notification(containers, f"질문: {question}")
    agent_stream = coordinator.stream_async(f"Provide a comprehensive analysis of: {question}")
    result = await show_streams(agent_stream, containers)
    logger.info(f"coordinator result: {result}")

    if chat.debug_mode == 'Enable':
        containers['status'].info(get_status_msg(f"end)"))

    return result
