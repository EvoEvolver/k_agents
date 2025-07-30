import random

from mllm import Chat
from mllm.utils.parser import Parse

from k_agents.agent_group.w_memory import WorkingMemory
from k_agents.translation.agent import TranslationAgentGroup
from k_agents.translation.code_translation import ExpCodeTranslationAgent
from k_agents.translation.procedure_translation import ProcedureTranslationAgent, \
    get_experiment_name_for_procedure


class TranslationAgentGroupLongContext(TranslationAgentGroup):

    def recall(self, wm: WorkingMemory, n_recall_items=None):
        prompt = []
        # shuffle the agents to ensure randomness
        random_ordered_agents = random.sample(self.translation_agents.agents, len(self.translation_agents.agents))
        for agent in random_ordered_agents:
            if isinstance(agent, ExpCodeTranslationAgent):
                desc = agent.get_exp_description()
                prompt.append(f"<experiment>{desc}</experiment>")
            elif isinstance(agent, ProcedureTranslationAgent):
                desc = f"Experiment name and arguments: {get_experiment_name_for_procedure(agent.title)}", "The experiment implements: " + agent.title + "\n" + "Steps: " + agent.steps + "\n" + "Background: " + agent.background
                prompt.append(f"<experiment>{desc}</experiment>")

        prompt = "\n".join(prompt)
        wm.add_content(prompt, tag="experiments")

    def codegen(self, wm: WorkingMemory, recall_res=None) -> str:
        experiments = wm.extract_tag_contents("experiments")[0]
        available_variables = wm.extract_tag_contents("available_variables")
        if len(available_variables) == 0:
            available_variables = "There is no available variables"
        else:
            available_variables = available_variables[0]
        instruction = wm.extract_tag_contents("instruction")[0]

        prompt = f"""
        Your task is to generate new code for the context described below.        
        <context>
        <available_variables>
        {available_variables}
        </available_variables>
        <instruction> 
        {instruction}
        </instruction>
        You must use the following experiments to generate the code:
        <experiments>
        {experiments}
        </experiments>
        </context>
        <requirements>
        You are required to generate code that can be used to implement <instruction> from the <experiment>.
        - The adopted code should absolutely just be what should implement <instruction>. 
        - Some of the <experiment> is irrelevant. But you should pick the most relevant one.
        You should first output an analysis of which experiment should be used to implement <instruction>.
        Then, wrapped by ```python and ```, output the new code that can implement <instruction>. The last line of the generated code must be in the format: `experiment_<ExperimentName> = <ExperimentName>(argument1,argument2, ...)`. The code must be executable. No placeholders are allowed. No import statements are allowed.
        </requirements>
        """
        chat = Chat(prompt, dedent=True)
        code = chat.complete(parse="quotes", expensive=True)
        if code.startswith("```"):
            code = Parse.quotes(code)
        return code