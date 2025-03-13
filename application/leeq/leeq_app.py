import mllm.config

from application.leeq.simulated_setup_2 import get_virtual_qubit_pair
from simulated_setup import *
from leeq.utils.ai.translation_agent import init_leeq_translation_agents
import streamlit as st
from k_agents.translation.env import TranslationAgentEnv
from k_agents.app import k_agents_app
import dotenv
dotenv.load_dotenv()

mllm.config.default_models.expensive = "gpt-4o"
mllm.config.default_models.normal = "gpt-4o"

st.set_page_config(page_title="K-agents for leeq", page_icon="🧠⚛️", layout="wide")

def initialize():
    if "agent_env" not in st.session_state:
        init_leeq_translation_agents()
        env = TranslationAgentEnv()
        st.session_state["agent_env"] = env

    if "variables" not in st.session_state:
        qubit_1, qubit_2 = get_virtual_qubit_pair()
        ExperimentManager().status().set_param("Plot_Result_In_Jupyter", False)
        st.session_state["variables"] = {"dut_1": qubit_1, "dut_2": qubit_2, "duts": (qubit_1, qubit_2)}

    if "suggested_procedure" not in st.session_state:
        st.session_state["suggested_procedure"] = "Two level Two-qubit calibration on `duts`"

k_agents_app(initialize)

