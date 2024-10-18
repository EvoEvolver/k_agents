from mllm import Chat
from mllm.utils.parser import Parse

from k_agents.memory.w_memory import WorkingMemory
from k_agents.translation.agent import TranslationAgentGroup
from k_agents.translation.code_translation import ExpCodeTranslationAgent
from k_agents.translation.procedure_translation import ProcedureTranslationAgent, \
    get_experiment_name_for_procedure


class TranslationAgentGroupRAG(TranslationAgentGroup):

    def recall(self, wm: WorkingMemory, n_recall_items=None):
        if n_recall_items is None:
            n_recall_items = self.n_recall_items
        ideas_from_score = self.translation_agents.get_ideas_by_score(wm, n_recall_items,
                                                                      None)
        prompt = []
        for idea in ideas_from_score:
            if isinstance(idea, ExpCodeTranslationAgent):
                desc = idea.get_exp_description()
                prompt.append(f"<experiment>{desc}</experiment>")
            elif isinstance(idea, ProcedureTranslationAgent):
                desc = f"Experiment name and arguments: {get_experiment_name_for_procedure(idea.title)}", "The experiment implements: " + idea.title + "\n" + "Steps: " + idea.steps + "\n" + "Background: " + idea.background
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
        <code_to_complete> 
        # [slot: {instruction}]
        </code_to_complete>
        You must use the following experiments to generate the code:
        {experiments}
        </context>
        <requirements>
        You are required to generate code that can be used to replace the <code_to_complete> from the <experiment>.
        - The adopted code should absolutely just be what should appear in the place of # [slot]. 
        - Some of the <experiment> is irrelevant. But you should pick the most relevant one.
        You should first output an analysis of which experiment should be used to fill the slot in <code_to_complete>.
        Then, wrapped by ```python and ```, output the new code that can fill the slot in <code_to_complete>. The last line of the generated code must be in the format: `experiment_<ExperimentName> = <ExperimentName>(argument1,argument2, ...)`. The code must be executable. No placeholders are allowed. No import statements are allowed.
        </requirements>
        """
        chat = Chat(prompt, dedent=True)
        code = chat.complete(parse="quotes", expensive=True)
        if code.startswith("```"):
            code = Parse.quotes(code)
        return code

class TranslationAgentRagJSON(TranslationAgentGroupRAG):
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
        <code_to_complete> 
        # [slot: {instruction}]
        </code_to_complete>
        You must use the following experiments to generate the code:
        {experiments}
        </context>
        <requirements>
        You are required to generate code that can be used to replace the <code_to_complete> from the <experiment>.
        - The adopted code should absolutely just be what should appear in the place of # [slot]. 
        - Some of the <experiment> is irrelevant. But you should pick the most relevant one.
        Output a JSON dict with the following keys:
        "analysis" (string): an ana lysis of which experiment should be used to fill the slot in <code_to_complete>.
        "code" (string): the new code that can fill the slot in <code_to_complete>. The last line of the generated code must be in the format: `experiment_<ExperimentName> = <ExperimentName>(argument1,argument2, ...)`. The code must be executable. No placeholders are allowed. No import statements are allowed.
        </requirements>
        """
        chat = Chat(prompt, dedent=True)
        res = chat.complete(parse="dict", expensive=True)
        code = res["code"]
        return code
