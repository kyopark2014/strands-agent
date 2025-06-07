import streamlit as st 
import chat
import utils
import json
import os

import logging
import sys

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
        "Stands Agent SDK를 이용하여 다양한 형태의 Agent를 구현합니다." 
        "상세한 코드는 [Github](https://github.com/kyopark2014/strands-agent)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["Agent", "Agent (Chat)"], index=0
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    strands_tools = ["calculator", "current_time", "python_repl", "use_aws"]
    mcp_tools = ["AWS documentation", "Wikipedia", "aws_cli"]
    mcp_options = strands_tools + mcp_tools

    mcp_selections = {}
    default_selections = ["current_time", "python_repl", "aws_cli"]

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

    # Get selected strands_tools from mcp_selections
    selected_strands_tools = [tool for tool in strands_tools if mcp_selections.get(tool, False)]
    selected_mcp_tools = [tool for tool in mcp_tools if mcp_selections.get(tool, False)]

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

    chat.update(modelName, reasoningMode, debugMode, selected_strands_tools, selected_mcp_tools)
    
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
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

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
    
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()
            
# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")

    with st.chat_message("assistant"):
        if mode == 'Agent':
            sessionState = ""
            chat.references = []
            chat.image_url = []
            response = chat.run_strands_agent(prompt, "Disable", st)

        elif mode == 'Agent (Chat)':
            sessionState = ""
            chat.references = []
            chat.image_url = []
            response = chat.run_strands_agent(prompt, "Enable", st)

    st.session_state.messages.append({"role": "assistant", "content": response})
    