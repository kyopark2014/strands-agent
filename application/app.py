import streamlit as st 
import chat
import json
import os
import mcp_config 
import asyncio
import logging
import sys
import knowledge_base as kb
import strands_agent

from langchain.docstore.document import Document

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

os.environ["DEV"] = "true"  # Skip user confirmation of get_user_input

# title
st.set_page_config(page_title='Strands Agent', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "Agent": [
        "Strands Agent SDK를 활용한 Agent를 이용합니다."
    ],
    "Agent (Chat)": [
        "대화가 가능한 Strands Agent입니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Stands Agent SDK를 이용하여 효율적인 Agent를 구현합니다." 
        "상세한 코드는 [Github](https://github.com/kyopark2014/strands-agent)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["Agent", "Agent (Chat)"], index=0
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    strands_tools = ["calculator", "current_time"]
    mcp_tools = [
        "basic", "code interpreter", "aws document", "aws cost", "aws cli", 
        "use_aws", "aws cloudwatch", "aws storage", "image generation", "aws diagram",
        "knowledge base", "perplexity", "ArXiv", "wikipedia", 
        "filesystem", "terminal", "text editor", "context7", "puppeteer", 
        "playwright", "firecrawl", "obsidian", "airbnb", 
        "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual", "사용자 설정"
    ]

    mcp_options = strands_tools + mcp_tools
    
    mcp_selections = {}
    default_selections = ["basic", "filesystem", "use_aws"]

    with st.expander("MCP 옵션 선택", expanded=True):            
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Split options into two groups
        mid_point = len(mcp_options) // 2
        first_half = mcp_options[:mid_point]
        second_half = mcp_options[mid_point:]
        
        # Display first group in the first column
        with col1:
            for option in first_half:
                default_value = option in default_selections
                mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
        
        # Display second group in the second column
        with col2:
            for option in second_half:
                default_value = option in default_selections
                mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)

    if mcp_selections["사용자 설정"]:
        mcp_info = st.text_area(
            "MCP 설정을 JSON 형식으로 입력하세요",
            value=mcp_config.mcp_user_config,
            height=150
        )
        logger.info(f"mcp_info: {mcp_info}")

        if mcp_info:
            mcp_config.mcp_user_config = json.loads(mcp_info)
            logger.info(f"mcp_user_config: {mcp_config.mcp_user_config}")
    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Claude 4 Opus', 'Claude 4 Sonnet', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=3
    )

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    
    # extended thinking of claude 3.7 sonnet
    select_reasoning = st.checkbox('Reasoning', value=False)
    reasoningMode = 'Enable' if select_reasoning else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    uploaded_file = None
    if mode=="Agent" or mode=="Agent (Chat)":
        st.subheader("📋 문서 업로드")
        # print('fileId: ', chat.fileId)
        uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "txt", "py", "md", "csv", "json"], key=chat.fileId)
    
    mcp_servers = [server for server, is_selected in mcp_selections.items() if is_selected]

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # print('clear_button: ', clear_button)

st.title('🔮 '+ mode)  

if clear_button==True:
    chat.initiate()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages():
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)            

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []     
    uploaded_file = None   
    
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()

file_name = ""
if uploaded_file is not None and clear_button==False:
    logger.info(f"uploaded_file.name: {uploaded_file.name}")
    if uploaded_file.name:
        logger.info(f"csv type? {uploaded_file.name.lower().endswith(('.csv'))}")

    if uploaded_file.name:
        chat.initiate()

        if debugMode=='Enable':
            status = '선택한 파일을 업로드합니다.'
            logger.info(f"status: {status}")
            st.info(status)

        file_name = uploaded_file.name
        logger.info(f"uploading... file_name: {file_name}")
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"file_url: {file_url}")

        kb.sync_data_source()  # sync uploaded files
            
        status = f'선택한 "{file_name}"의 내용을 요약합니다.'
        if debugMode=='Enable':
            logger.info(f"status: {status}")
            st.info(status)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        logger.info(f"msg: {msg}")

        st.write(msg)

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")
    #logger.info(f"is_updated: {agent.is_updated}")

    with st.chat_message("assistant"):
        if mode == 'Agent':
            history_mode = "Disable"
        elif mode == 'Agent (Chat)':
            history_mode = "Enable"

        containers = {
            "tools": st.empty(),
            "status": st.empty(),
            "notification": [st.empty() for _ in range(100)],
            "key": st.empty()
        }
        
        response, image_urls = asyncio.run(strands_agent.run_agent(prompt, strands_tools, mcp_servers, history_mode, containers))

        if chat.debug_mode == 'Disable':
           st.markdown(response)
        
        for url in image_urls:
            logger.info(f"url: {url}")
            file_name = url[url.rfind('/')+1:]
            st.image(url, caption=file_name, use_container_width=True)      

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": image_urls if image_urls else []
        })
    