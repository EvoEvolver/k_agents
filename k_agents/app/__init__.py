import streamlit as st
from k_agents.io_interface import set_streamlit_display_impl
from k_agents.app.pages import draw_runner
from k_agents.app.pages import draw_knowledge_page
import litellm

def k_agents_app():
    if st.session_state["agent_env"] is None:
        st.write("Please specify `agent_env` in the session state.")
        st.stop()
    if st.session_state["variables"] is None:
        st.write("Please specify `variables` in the session state.")
        st.stop()

    set_streamlit_display_impl()
    st.sidebar.title("K-agents")
    page = st.sidebar.radio("pages", ["Knowledge Base", "Executor"])
    if page == "Knowledge Base":
        draw_knowledge_page()
    elif page == "Executor":
        draw_runner()

