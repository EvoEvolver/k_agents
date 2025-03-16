import streamlit as st
from k_agents.io_interface import set_streamlit_display_impl
from k_agents.app.pages import draw_runner
from k_agents.app.pages import draw_knowledge_page
import litellm

def k_agents_app(initialize):
    set_streamlit_display_impl()
    if "agent_env" not in st.session_state:
        st.write("Initializing")
        initialize()
        st.write("Initialization done.")
    if "variables" not in st.session_state:
        st.write("Please specify `variables` in the session state.")
        st.stop()

    st.sidebar.title("K-agents")
    page = st.sidebar.radio("pages", ["Executor", "Knowledge Base"])
    if page == "Executor":
        draw_runner()
    if page == "Knowledge Base":
        draw_knowledge_page()

