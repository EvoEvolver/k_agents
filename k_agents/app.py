from k_agents.io_interface import set_streamlit_display_impl
import streamlit as st
import os
import openai
from k_agents.execution.agent import execute_procedure
from application.example_lab import example_lab
from k_agents.translation.agent import init_translation_agents
from mllm.config import default_models
default_models.normal = "gpt-4o"
default_models.expensive = "gpt-4o"


set_streamlit_display_impl()

def check_openai_key(api_key: str) -> bool:
    openai.api_key = api_key
    try:
        openai.models.list()
        return True
    except Exception as e:
        return False

with st.sidebar:
    st.title('K-agents')
    # add a text input widget
    openai_api_key = st.text_input("OpenAI API key")

    required_output = st.text_area(label="Experiment procedure", value="""
    - Generate Random Lines with secret_parameter being 2. If failed, retry 4 times.
    - Generate Random Dots
    """.strip())

    button = st.button("Run")

if button:
    if not openai_api_key:
        if os.environ.get("OPENAI_API_KEY") is not None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            st.write("Using OpenAI API key from environment variable")
        else:
            st.write("Please provide OpenAI API key")
            st.stop()
    else:
        if not check_openai_key(openai_api_key):
            st.write("Invalid OpenAI API key")
            st.stop()

    init_translation_agents(example_lab)

    execute_procedure(required_output)