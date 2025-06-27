import streamlit as st
import boto3
from botocore.exceptions import ClientError
import base64
from io import BytesIO
from PIL import Image
import time
from utils import load_env_variables, get_bedrock_client, check_aws_credentials

# 환경 변수 로드
load_env_variables()

# 페이지 설정
st.set_page_config(
    page_title="Claude Sonnet 챗봇",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# AWS Bedrock 클라이언트 설정
@st.cache_resource
def get_cached_bedrock_client():
    return get_bedrock_client()

# Claude Sonnet 모델 ID
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# 시스템 프롬프트 설정
SYSTEM_PROMPT = "당신은 친절하고 도움이 되는 AI 비서입니다. 사용자의 질문에 명확하고 정확하게 답변해 주세요."

# 텍스트 메시지 처리 함수
def process_text_message(client, user_input):
    """
    텍스트 메시지를 처리하는 함수
    """
    # 사용자 메시지 생성
    user_message = {
        "role": "user",
        "content": [{"text": user_input}]
    }
    
    # 대화 기록에 사용자 메시지 추가
    st.session_state.conversation_history.append(user_message)
    
    # 시스템 프롬프트 설정
    system_prompts = [{"text": SYSTEM_PROMPT}]
    
    # 추론 파라미터 설정
    inference_config = {"temperature": 0.7}
    additional_model_fields = {"top_k": 250}
    
    try:
        # Converse API 호출
        response = client.converse(
            modelId=MODEL_ID,
            messages=st.session_state.conversation_history,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )
        
        # 모델 응답 추출
        output_message = response['output']['message']
        
        # 대화 기록에 모델 응답 추가
        st.session_state.conversation_history.append(output_message)
        
        # 응답 텍스트 추출
        response_text = ""
        for content in output_message['content']:
            if 'text' in content:
                response_text += content['text']
        
        return response_text
        
    except ClientError as err:
        message = err.response['Error']['Message']
        st.error(f"오류 발생: {message}")
        return f"오류가 발생했습니다: {message}"

# 스트리밍 응답 처리 함수
def process_text_message_streaming(client, user_input):
    """
    텍스트 메시지를 처리하고 응답을 스트리밍하는 함수
    """
    # 사용자 메시지 생성
    user_message = {
        "role": "user",
        "content": [{"text": user_input}]
    }
    
    # 대화 기록에 사용자 메시지 추가
    st.session_state.conversation_history.append(user_message)
    
    # 시스템 프롬프트 설정
    system_prompts = [{"text": SYSTEM_PROMPT}]
    
    # 추론 파라미터 설정
    inference_config = {"temperature": 0.7}
    additional_model_fields = {"top_k": 250}
    
    try:
        # ConverseStream API 호출
        response = client.converse_stream(
            modelId=MODEL_ID,
            messages=st.session_state.conversation_history,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )
        
        # 스트리밍 응답 처리
        stream = response.get('stream')
        full_response = ""
        
        if stream:
            for event in stream:
                if 'contentBlockDelta' in event:
                    text_chunk = event['contentBlockDelta']['delta']['text']
                    full_response += text_chunk
                    # 실시간으로 응답 업데이트
                    yield full_response
        
        # 대화 기록에 모델 응답 추가
        assistant_message = {
            "role": "assistant",
            "content": [{"text": full_response}]
        }
        st.session_state.conversation_history.append(assistant_message)
        
        return full_response
        
    except ClientError as err:
        message = err.response['Error']['Message']
        st.error(f"오류 발생: {message}")
        yield f"오류가 발생했습니다: {message}"

# 이미지 메시지 처리 함수
def process_image_message(client, user_input, image):
    """
    이미지와 텍스트를 함께 처리하는 함수
    """
    # 이미지를 바이트로 변환
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    
    # 사용자 메시지 생성
    user_message = {
        "role": "user",
        "content": [
            {"text": user_input},
            {
                "image": {
                    "format": "png",
                    "source": {
                        "bytes": image_bytes
                    }
                }
            }
        ]
    }
    
    # 대화 기록에 사용자 메시지 추가
    st.session_state.conversation_history.append(user_message)
    
    # 시스템 프롬프트 설정
    system_prompts = [{"text": SYSTEM_PROMPT}]
    
    try:
        # Converse API 호출
        response = client.converse(
            modelId=MODEL_ID,
            messages=st.session_state.conversation_history,
            system=system_prompts
        )
        
        # 모델 응답 추출
        output_message = response['output']['message']
        
        # 대화 기록에 모델 응답 추가
        st.session_state.conversation_history.append(output_message)
        
        # 응답 텍스트 추출
        response_text = ""
        for content in output_message['content']:
            if 'text' in content:
                response_text += content['text']
        
        return response_text
        
    except ClientError as err:
        message = err.response['Error']['Message']
        st.error(f"오류 발생: {message}")
        return f"오류가 발생했습니다: {message}"

# 문서 처리 함수
def process_document_message(client, user_input, document):
    """
    문서와 텍스트를 함께 처리하는 함수
    """
    # 문서 형식 가져오기 및 바이트로 읽기
    document_format = document.name.split(".")[-1]
    document_bytes = document.getvalue()
    
    # 사용자 메시지 생성
    user_message = {
        "role": "user",
        "content": [
            {"text": user_input},
            {
                "document": {
                    "name": document.name,
                    "format": document_format,
                    "source": {
                        "bytes": document_bytes
                    }
                }
            }
        ]
    }
    
    # 대화 기록에 사용자 메시지 추가
    st.session_state.conversation_history.append(user_message)
    
    # 시스템 프롬프트 설정
    system_prompts = [{"text": SYSTEM_PROMPT}]
    
    try:
        # Converse API 호출
        response = client.converse(
            modelId=MODEL_ID,
            messages=st.session_state.conversation_history,
            system=system_prompts
        )
        
        # 모델 응답 추출
        output_message = response['output']['message']
        
        # 대화 기록에 모델 응답 추가
        st.session_state.conversation_history.append(output_message)
        
        # 응답 텍스트 추출
        response_text = ""
        for content in output_message['content']:
            if 'text' in content:
                response_text += content['text']
        
        return response_text
        
    except ClientError as err:
        message = err.response['Error']['Message']
        st.error(f"오류 발생: {message}")
        return f"오류가 발생했습니다: {message}"

# AWS 자격 증명 확인
aws_credentials_valid = check_aws_credentials()

# Streamlit UI 구성
st.title("🤖 Claude Sonnet 챗봇")
st.markdown("AWS Bedrock의 Claude Sonnet 모델을 활용한 챗봇입니다. 질문을 입력하거나 이미지/문서를 업로드하세요.")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # 스트리밍 모드 설정
    streaming_mode = st.checkbox("스트리밍 응답 활성화", value=True)
    
    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.success("대화가 초기화되었습니다.")
    
    st.markdown("---")
    st.markdown("### 정보")
    st.markdown("이 애플리케이션은 AWS Bedrock의 Claude Sonnet 모델을 사용합니다.")
    st.markdown("Anthropic의 Claude 모델은 텍스트, 이미지, 문서를 처리할 수 있습니다.")

# AWS 자격 증명이 유효하지 않은 경우 경고 표시
if not aws_credentials_valid:
    st.warning("AWS 자격 증명이 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.info("1. .env.example 파일을 .env로 복사하세요.\n2. AWS 자격 증명을 입력하세요.")
    st.stop()

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 탭 설정
tab1, tab2 = st.tabs(["이미지 업로드", "문서 업로드"])

# 이미지 업로드 탭
with tab1:
    uploaded_image = st.file_uploader("이미지 업로드 (선택사항)", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(Image.open(uploaded_image), caption="업로드된 이미지", width=400)

# 문서 업로드 탭
with tab2:
    uploaded_document = st.file_uploader("문서 업로드 (선택사항)", type=["pdf", "txt", "docx"])
    if uploaded_document is not None:
        st.write(f"업로드된 문서: {uploaded_document.name}")

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지 표시
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 로딩 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Bedrock 클라이언트 가져오기
        client = get_cached_bedrock_client()
        
        # 이미지가 업로드된 경우
        if uploaded_image is not None:
            message_placeholder.markdown("이미지를 분석 중...")
            response = process_image_message(client, prompt, Image.open(uploaded_image))
            message_placeholder.markdown(response)
        
        # 문서가 업로드된 경우
        elif uploaded_document is not None:
            message_placeholder.markdown("문서를 분석 중...")
            response = process_document_message(client, prompt, uploaded_document)
            message_placeholder.markdown(response)
        
        # 텍스트만 있는 경우
        else:
            # 스트리밍 모드가 활성화된 경우
            if streaming_mode:
                response = ""
                for response_chunk in process_text_message_streaming(client, prompt):
                    message_placeholder.markdown(response_chunk + "▌")
                    time.sleep(0.01)
                message_placeholder.markdown(response_chunk)
                response = response_chunk
            # 일반 모드
            else:
                message_placeholder.markdown("생각 중...")
                response = process_text_message(client, prompt)
                message_placeholder.markdown(response)
    
    # 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})