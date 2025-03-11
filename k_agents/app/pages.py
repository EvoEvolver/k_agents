import inspect
import os

import litellm
import openai
import streamlit as st

from k_agents.execution.agent import execute_procedure
from k_agents.inspection.agents import TextInspectionAgent, VisualInspectionAgent
from k_agents.translation.code_translation import ExpCodeTranslationAgent
from k_agents.translation.env import TranslationAgentEnv
from k_agents.translation.procedure_translation import ProcedureTranslationAgent
from mllm.config import default_models

default_models.normal = "gpt-4o"
default_models.expensive = "gpt-4o"



def slugify(text):
    return text.lower().replace(" ", "-")

def draw_knowledge_page():
    st.write("# Knowledge base")

    env = st.session_state["agent_env"]
    agents = env.translation_agents.translation_agents.agents

    display_experiments = st.button("Show experiments")
    display_procedures = st.button("Show procedures")

    if display_experiments:

        st.write("## Available experiments")
        # toc
        for agent in agents:
            if isinstance(agent, ExpCodeTranslationAgent):
                st.write(f"- [{agent.exp_name}](#{slugify(agent.exp_name)})")

        for agent in agents:
            if isinstance(agent, ExpCodeTranslationAgent):
                st.write(f"# {agent.exp_name}")
                exp_cls = agent.exp_cls
                run_method = exp_cls.run
                signature = inspect.signature(run_method)
                # remove the first argument
                signature = signature.replace(
                    parameters=list(signature.parameters.values())[1:])
                doc = inspect.getdoc(run_method)
                st.code(f"""
    def {run_method.__name__}{signature}:
        '''
        {doc}
        '''
        ...
    """.strip())
                for ins_agent in exp_cls._get_inspection_agents():

                    if isinstance(ins_agent, TextInspectionAgent):
                        st.write(f"## Text inspection: {ins_agent.func.__name__}")
                        st.code(inspect.getsource(ins_agent.func))
                    if isinstance(ins_agent, VisualInspectionAgent):
                        st.write(f"## Visual inspection: {ins_agent.plot_func.__name__}")
                        st.write(ins_agent.prompt)



    if display_procedures:
        st.write("# Available Procedures")
        for agent in agents:
            if isinstance(agent, ProcedureTranslationAgent):
                st.write(f"# {agent.title}")
                st.write(agent.steps)


def check_openai_key(api_key: str) -> bool:
    openai.api_key = api_key
    try:
        openai.models.list()
        return True
    except Exception as e:
        return False


def draw_runner():
    with st.sidebar:
        st.title('Executor')

        if os.environ.get("OPENAI_API_KEY") is None:
            # add a text input widget
            openai_api_key = st.text_input("OpenAI API key")
        else:
            openai_api_key = os.environ.get("OPENAI_API_KEY")

        if "suggested_procedure" in st.session_state:
            suggested_procedure = st.session_state["suggested_procedure"]
        else:
            suggested_procedure = ""
        required_output = st.text_area(label="Experiment procedure", value=suggested_procedure)

        variables = st.session_state["variables"]
        if len(variables) != 0:
            st.write("Variables:")
        for key, value in variables.items():
            st.write(f"`{key}`: `{value}`")


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
        litellm.api_key = openai_api_key
        if TranslationAgentEnv().translation_agents is None:
            st.write("Translation agents not initialized")
            st.stop()

        variables = st.session_state["variables"]

        execute_procedure(required_output, **variables)
