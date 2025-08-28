import streamlit as st
import os
import re
from datetime import datetime
from typing import Dict, List
import json

# Import your existing chatbot class
from chatbot import UIUCChatBot
from config import Config

# UIUC Color Palette - Minimal
UIUC_BLUE = "#13294B" 
UIUC_LIGHT_BLUE = "#E3F2FD"
UIUC_LIGHT_ORANGE = "#ffb38a"  # Better orange color

# Page configuration
st.set_page_config(
    page_title="UIUC CS Assistant",
    page_icon="🐻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal styling - clean chatbot interface
st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 8rem;
        max-width: 700px;
    }}
    
    .chat-container {{
        margin-bottom: 8rem;
        padding-bottom: 3rem;
        max-height: calc(100vh - 200px);
        overflow-y: auto;
    }}
    
    .stChatInput {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #eee;
    }}
    
    .header-container {{
        text-align: center;
        padding: 1rem 0 1.5rem 0;
        margin-bottom: 1rem;
    }}
    
    .illinois-logo {{
        font-size: 1.8rem;
        color: {UIUC_BLUE};
        font-weight: bold;
        margin-bottom: 0.3rem;
    }}
    
    .subtitle {{
        color: {UIUC_BLUE};
        font-size: 0.9rem;
        margin: 0;
        opacity: 0.8;
    }}
    
    .user-message {{
        background-color: {UIUC_LIGHT_BLUE};
        padding: 0.8rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0 0.5rem 3rem;
        border: none;
        font-size: 0.9rem;
    }}
    
    .assistant-message {{
        background-color: {UIUC_LIGHT_ORANGE};
        padding: 0.8rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 3rem 0.5rem 0;
        border: none;
        font-size: 0.9rem;
    }}
    
    .stChatInput > div > div > textarea {{
        border-radius: 20px;
        border: 1px solid #ddd;
        font-size: 0.9rem;
    }}
    
    .stChatInput > div > div > textarea:focus {{
        border-color: {UIUC_BLUE};
        box-shadow: 0 0 0 1px {UIUC_BLUE};
    }}
    
    /* Hide Streamlit elements */
    .stDeployButton {{display: none;}}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Sidebar styling */
    .css-1d391kg {{background-color: #f8f9fa;}}
    
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

def initialize_chatbot():
    """Initialize the RAG chatbot"""
    if st.session_state.chatbot is None:
        # Check if API keys are available
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
            st.error("❌ Please set OPENAI_API_KEY and PINECONE_API_KEY in your .env file")
            st.stop()
        
        with st.spinner("🤔 Loading course information..."):
            try:
                st.session_state.chatbot = UIUCChatBot()
                return True
            except Exception as e:
                st.error(f"❌ Unable to load course data: {str(e)}")
                st.stop()
    return True

def create_new_session():
    """Create a new chat session"""
    session_id = f"session_{len(st.session_state.chat_sessions) + 1}_{datetime.now().strftime('%H%M%S')}"
    st.session_state.chat_sessions[session_id] = {
        'name': f"Chat {len(st.session_state.chat_sessions) + 1}",
        'messages': [],
        'created_at': datetime.now()
    }
    st.session_state.current_session = session_id
    return session_id

def display_chat_message(message: Dict, is_user: bool = False):
    """Display a clean chat message"""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Clean the content - remove HTML tags and divs that might leak through
        clean_content = re.sub(r'<[^>]+>', '', message['content'])
        clean_content = re.sub(r'<div[^>]*>', '', clean_content)
        clean_content = re.sub(r'</div>', '', clean_content)
        
        st.markdown(f"""
        <div class="assistant-message">
            {clean_content}
        </div>
        """, unsafe_allow_html=True)

def sidebar():
    """Minimal sidebar for session management only"""
    with st.sidebar:
        st.markdown("### 💬 Chat History")
        
        if st.button("⊕ New Chat", use_container_width=True):
            create_new_session()
            st.rerun()
        
        # List existing sessions
        if st.session_state.chat_sessions:
            for session_id, session_data in st.session_state.chat_sessions.items():
                is_current = session_id == st.session_state.current_session
                
                if is_current:
                    st.markdown(f"**▶ {session_data['name']} (current)**")
                else:
                    if st.button(f"○ {session_data['name']}", 
                               key=f"session_{session_id}",
                               use_container_width=True):
                        st.session_state.current_session = session_id
                        st.rerun()
        
        st.divider()
        
        # Clear all chats
        if st.button("⌫ Clear All", use_container_width=True):
            st.session_state.chat_sessions = {}
            st.session_state.current_session = None
            st.rerun()

def main_chat_interface():
    """Clean minimal chat interface"""
    # Simple header with Illinois logo
    st.markdown("""
    <div class="header-container">
        <div class="illinois-logo">🐻 UIUC CS Assistant</div>
        <p class="subtitle">Ask about computer science courses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensure we have a current session
    if not st.session_state.current_session:
        create_new_session()
    
    current_session = st.session_state.chat_sessions[st.session_state.current_session]
    
    # Display chat history in container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in current_session['messages']:
            display_chat_message(message, is_user=(message['type'] == 'user'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input at bottom
    user_input = st.chat_input("Ask about courses, prerequisites, difficulty...")
    
    if user_input:
        # Add user message
        current_session['messages'].append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Get bot response
        with st.spinner("🤔"):
            try:
                result = st.session_state.chatbot.query(user_input)
                
                current_session['messages'].append({
                    'type': 'bot',
                    'content': result['answer'],
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                current_session['messages'].append({
                    'type': 'bot',
                    'content': f"Sorry, I encountered an error: {str(e)}",
                    'timestamp': datetime.now()
                })
        
        st.rerun()

def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Initialize chatbot
    initialize_chatbot()
    
    # Show sidebar
    sidebar()
    
    # Show main interface
    main_chat_interface()

if __name__ == "__main__":
    main()