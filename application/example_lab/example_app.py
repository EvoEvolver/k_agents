import streamlit as st
from k_agents.translation.agent import init_translation_agents
from application.example_lab import example_lab
from k_agents.translation.env import TranslationAgentEnv
from k_agents.app import k_agents_app
import dotenv
dotenv.load_dotenv()
st.set_page_config(page_title="K-agents example", page_icon="🧠", layout="wide")

init_translation_agents(example_lab)
env = TranslationAgentEnv()
st.session_state["agent_env"] = env
st.session_state["variables"] = {}
k_agents_app()

