import re
import hashlib
import tempfile
import shutil
import warnings

# Suppress matplotlib warnings for non-interactive backend
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import streamlit as st
# st.set_page_config(
#     page_title="Sales Pulse",
#     page_icon="🥚",
#     layout="wide",
#     initial_sidebar_state="expanded"
#     )

st.set_page_config(
    page_title="IKEA Store Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import time
import matplotlib
# Configure matplotlib for Streamlit (non-interactive backend)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from constants.constant import *
from tools.tools_agents import *
from data.dataloader import vector_db
from configuration.config import *
from main import process_query
from PIL import Image
import pandas as pd
from utils.utils import create_citations, image_to_base64
from logger import logger
from langchain_experimental.utilities import PythonREPL

# Set environment variables
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Define app pages
PAGE_LOGIN = "login"
PAGE_CHAT = "chat"

repl = PythonREPL()
class BadRequestError(Exception):
    pass  

def initialize_session_state():
    if "page" not in st.session_state:
        st.session_state.page = PAGE_LOGIN
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = 0000000000
    if 'generated_plots' not in st.session_state:
        st.session_state.generated_plots = []


def clear_temp(dirname:str):
    """helper function to clear the session temp"""
    # delete the artifact dirctory
    if os.path.exists(dirname):
        try:
            shutil.rmtree(path=dirname)
            logger.info(f"✅ tempdir cleaned {os.path.basename(dirname)}")
        except Exception as e:
            logger.error(f"❌ tempdir deletion failed {os.path.basename(dirname)}")
            pass


def authenticate(username, password):
    # Simple authentication - In a real app, use secure authentication
    # This is just for demonstration
    valid_credentials = {
        "Admin": "admin123",
        "Jane Doe": "user123",
        "John Doe": "user123",
        "Demo User": "demo123"
    }
    
    if username in valid_credentials and password == valid_credentials[username]:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.page = PAGE_CHAT
        st.session_state.session_id = generate_session_id()
        return True
    else:
        st.session_state.login_attempts += 1
        return False

def logout():
    """logout button handler"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.page = PAGE_LOGIN
    st.session_state.messages = []
    st.session_state.session_id = 0000000000
    st.session_state.generated_plots = []
    st.rerun()

def refresh_chat():
    """chat refresh handler"""
    st.session_state.messages = []
    st.session_state.generated_plots = []
    st.rerun()

def render_login_page():
    """creates the login page"""
    base_dir = os.getcwd()
    image_path = os.path.join(base_dir, 'Nihilent_logo.gif')
    nihilent_logo_img = image_to_base64(Image.open(image_path))

    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{nihilent_logo_img}" width="200" style="margin-bottom: 1rem;" />
                <div class="login-container">
                    <div class="login-header">
                        <h1 style="color: {PRIMARY_COLOR}; margin-bottom: 0.5rem;">🤖 Sales Pulse Assistant</h1>
                        <p style="color: #999; font-size: 14px;">Login to access your dashboard</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.container():
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            col1, col2 = st.columns([1, 1])
            with col1:
                login_button = st.button("Login", use_container_width=True)
            with col2:
                demo_button = st.button("Demo Login", use_container_width=True)
            
            if login_button:
                if authenticate(username, password):
                    st.success("Login successful! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Invalid credentials. Attempts: {st.session_state.login_attempts}")
                    
            if demo_button:
                authenticate("Demo User", "demo123")
                st.success("Demo login successful! Redirecting...")
                time.sleep(1)
                st.rerun()

def render_chat_page():
    """creates the chat page"""
    base_dir = os.getcwd()
    image_path = os.path.join(base_dir, 'Nihilent_logo.gif')
    nihilent_logo_img = Image.open(image_path)
    logger.info(f"⚠️ user {st.session_state.username} logged in; session id {st.session_state.session_id}")

    # create a temp directory for a user session
    tempdir_root = os.path.join(os.getcwd(),'./.tmp')
    os.makedirs(tempdir_root,exist_ok=True)
    # creates a hashex value for user session
    hex_salt = f"{st.session_state.username}_{st.session_state.session_id}"
    # hash hex of the session for the user
    hash_hex = hashlib.sha256(hex_salt.encode()).hexdigest()
    artifact_dir = os.path.join(tempdir_root,hash_hex)
    if os.path.exists(artifact_dir):
        logger.warning(f"⚠️ artifact dir already exists: {os.path.basename(artifact_dir)}")
    else:
        # create the temp directory
        logger.info(f"⚠️ tempdir created at => {artifact_dir}")
        os.makedirs(artifact_dir,exist_ok=True)
    
    # Sidebar 
    with st.sidebar:
        st.image(nihilent_logo_img, width=200)
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <span class="gray-text">IKEA Store Analytics Assistant</span>
                <p style="margin-top: 5px; color: #666;">AE Region • DJA & YAS Stores</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(f"<p style='text-align: center;'>Welcome, <b>{st.session_state.username}</b>!</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Chat", use_container_width=True):
                clear_temp(dirname=artifact_dir)
                refresh_chat()
        with col2:
            if st.button("Logout", use_container_width=True):
                clear_temp(dirname=artifact_dir)
                logout()
    
    # Main chat area
    st.markdown("<h1 style='text-align: center; color: #3366FF;'>Sales Pulse Assistant</h1>", unsafe_allow_html=True)
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Add space before the chat input
    st.markdown("<div style='padding-bottom: 10px;'></div>", unsafe_allow_html=True)
    # Chat input        
    if prompt := st.chat_input("🤖 Ask your questions"):
        logger.info(f"################### RECEIVED REQUEST ################################")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # prepare chat history for agent invocation
        chat_history = [{'role':message["role"],'content':message["content"]} for message in st.session_state.messages[-3:]]
        chat_history = " \n".join([f"{chat['role']}: {chat['content']}" for chat in chat_history])
        logger.info(f"⚙️ invoking agentic executor: {prompt}")
        try:
            with st.spinner("🤔 Thinking"):
                if prompt.lower().strip()=="generate dashboard":
                    logger.info(f"⚠️ generating dash app")
                    logger.info(f"🗨️ Text Messages found in user session: {len(st.session_state.messages)}")
                    logger.info(f"📊 Total plots found in user session: {len(st.session_state.generated_plots)}")
                    if len(st.session_state.generated_plots):
                        for i,plot in enumerate(st.session_state.generated_plots):
                            st.progress(i,text="🏃🏼generating plots for the dashboard; please wait")
                            # generate the plot and save it
                            repl.run(repl.sanitize_input(plot))
                else:
                    answer = process_query(prompt=prompt,chat_history=chat_history)
                    logger.info(f"✅ Response => {answer}")
                    if len(answer):
                        result_ = answer['output']
                        result = remove_markdown_images(result_)
                        logger.info(f"✅ Agent Response Processed => {result}")

                        # get the intermediate steps to create citations
                        intermediate_steps = answer["intermediate_steps"]
                        tool_usage = []
                        for steps in intermediate_steps:
                            tool_usage.append(steps[0].tool)
                        
                        # tool handler
                        sorted_list_of_dicts = []
                        if 'query_documents' in tool_usage:
                            results = vector_db.similarity_search_with_score(result)
                            list_of_dicts = []
                            for res in results:
                                score = res[1]
                                filename = res[0].metadata['source']
                                filename = os.path.basename(filename)
                                page_number = res[0].metadata['page'] + 1
                                dictionary = {"source": filename, "page": page_number, "score": score}
                                list_of_dicts.append(dictionary)
                            sorted_list_of_dicts = sorted(list_of_dicts, key=lambda x: x["score"])   
                        
                        if 'python_repl' in tool_usage:
                            for steps in intermediate_steps:
                                if steps[0].tool=='python_repl':
                                    plotting_code = steps[0].tool_input.get('code','')
                                    if check_for_charts(plotting_code):
                                        dashboard_prompt = f""" 
                                            Update the provided code so that it saves the resulting plot as a .jpeg file in the specified folder.

                                            Requirements:
                                                -Keep all variable names and data types exactly as they are.
                                                -Only make the minimal changes necessary to enable saving the plot as a .jpeg.
                                                -Do not add, remove, or modify any other part of the code.
                                                -Do not include any explanations or extra text. Return only the corrected code block.

                                            Inputs:
                                            Code:
                                            {plotting_code}

                                            Save location:
                                            {artifact_dir}

                                            Output:
                                            The updated code that saves the plot to {artifact_dir}/<provide a apt filename here>.jpeg
                                            Strict rule:
                                            Any deviation from these instructions, including extra explanations or changes to variable names,
                                            will result in a heavy penalty.
                                        """
                                        plotting_code_updated = llm.invoke(dashboard_prompt).content
                                        logger.info(f"✅ plotting code updated:\n {plotting_code_updated}")
                                        st.session_state.generated_plots.append(plotting_code_updated)

                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = result
                            message_placeholder.markdown(full_response)
                        
                        if sorted_list_of_dicts:
                            logger.info("📑 Citations =>", sorted_list_of_dicts)
                            # process the citations
                            cite_df = pd.DataFrame(sorted_list_of_dicts)
                            cite_df = cite_df.sort_values(by='score',ascending=False).reset_index(drop=True)
                            cite_df = cite_df.drop_duplicates(subset='source',keep='first').reset_index(drop=True)
                            cite_dict = cite_df.to_dict('records')
                            with st.container():
                                st.markdown("<p style='font-size: 14px; font-weight: bold; margin-bottom: 5px;'>Sources:</p>", unsafe_allow_html=True)
                                st.markdown("<ul style='margin-top: 0; padding-left: 20px;'>", unsafe_allow_html=True)
                                for item in cite_dict:
                                    cite_path = os.path.join(
                                        os.getcwd(),
                                        "data",
                                        "documents",
                                        os.path.basename(item['source'])
                                    )
                                    logger.info(f"🔗 Cite Document => {cite_path}")
                                    cite_url = create_citations(file_path=cite_path)
                                    logger.info(f"🔗 Cite URL => {cite_url}")
                                    # Create the source link with the local PDF path
                                    source_link = f"""<li><a href={cite_url}>🔗 {item['source']}; Page {item['page']} </a></li>"""
                                    st.markdown(source_link, unsafe_allow_html=True)     
                                st.markdown("</ul>", unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": full_response}) 
        except Exception as e:
                logger.critical(f"❌ Retry also failed: {e}")
                fail_note = "Apologies 😔, I couldn't process your request. Please reframe your query or try again later 👍🏼"
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown(fail_note, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": fail_note})

def main():
    """main function to render the streamlit ui"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    initialize_session_state()
    if st.session_state.page == PAGE_LOGIN and not st.session_state.authenticated:
        render_login_page()
    else:
        render_chat_page()

if __name__ == "__main__":
    main()