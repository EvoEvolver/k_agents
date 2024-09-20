from mllm import Chat
from mllm.utils.parser import Parse

from k_agents.ideanet.lt_memory import RecallResult
from k_agents.ideanet.w_memory import WorkingMemory
from k_agents.translation.agent import TranslationAgent
from k_agents.translation.code_indexer import ExperimentCodegenIdea


class TranslationAgentRAG(TranslationAgent):

    def recall(self, wm: WorkingMemory) -> RecallResult:
        ideas_from_score = self.lt_memory.get_ideas_by_score(wm, self.n_recall_items,
                                                             None)
        prompt = []
        for idea in ideas_from_score:
            idea: ExperimentCodegenIdea
            desc = idea.get_exp_description()
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

        notices = """
- Call exactly one time to the experiment function / class in this edit.
- Every class or function call will include the data analysis inside the call automatically so there is no need to do data analysis separately.
- Always use named arguments to call functions or classes.
- Store the return value of the call functions or classes to a variable.
"""
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
The last line of the generated code must be in the format: `experiment_<name> = <ExperimentName>(argument1,argument2, ...)`
Some of the <experiment> is irrelevant. But you should pick the most relevant one. You have to pick one of the experiment.

You should first output an analysis of which experiment should be used to fill the slot in <code_to_complete>.
Then, wrapped by ```python and ```, output the new code that can fill the slot in <code_to_complete>. The last line of the generated code must be in the format: `experiment_<name> = <ExperimentName>(argument1,argument2, ...)`. The code must be executable. 
</requirements>
"""
        chat = Chat(prompt)
        code = chat.complete(parse="quotes", expensive=True)  # ["code"]
        if code.startswith("```"):
            code = Parse.quotes(code)
        return code


"""
Output a JSON dict with the following keys:
"analysis" (string): an ana lysis of which experiment should be used to fill the slot in <code_to_complete>.
"code" (string): the new code that can fill the slot in <code_to_complete>. The last line of the generated code must be in the format: `experiment_<name> = <ExperimentName>(argument1,argument2, ...)`. The code must be executable.
"""
