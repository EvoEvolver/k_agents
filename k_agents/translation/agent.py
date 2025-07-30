import os
from typing import List, Type, Any, Callable

import numpy as np
from fibers.data_loader.module_to_tree import get_tree_for_module
from fibers.tree.node_attr.code import get_type, get_obj
from mllm import Chat
from mllm.utils import p_map
from mllm.utils.parser import Parse

from k_agents.experiment.experiment import Experiment
from k_agents.agent_group.code_wmemory import CodeWMemoryItem
from k_agents.agent_group.agent_group import RetrievableAgent, AgentResult, AgentGroup, AgentGroupResult
from k_agents.agent_group.w_memory import WorkingMemory, WMemoryItem
from k_agents.translation.code_translation import add_exp_to_agent_group, \
    add_exp_to_var_table
from k_agents.translation.env import TranslationAgentEnv
from k_agents.translation.procedure_translation import extract_procedures_to_agent_group
from k_agents.variable_table import VariableTable


class TranslationAgentGroup:

    def __init__(self):
        self.translation_agents = AgentGroup()
        self.codegen_agent = CodegenAgentUltraFiltered()
        self.n_recall_items = 10
        self._cached_recall_res = None

    def recall(self, wm: WorkingMemory, n_recall_items=None) -> AgentGroupResult:
        """
        Recall agents from long term memory, using what is currently in the working memory.

        :param wm: the working memory to stimuli agents
        :return: the result of triggered agents
        """
        if n_recall_items is None:
            n_recall_items = self.n_recall_items
        res = self.translation_agents.recall_by_wm(wm, top_k=n_recall_items)
        self._cached_recall_res = res
        return res

    def codegen(self, wm: WorkingMemory, recall_res: AgentGroupResult = None) -> str:
        """
        Generate code from working memory, updates working memory with recalled agents in the process.

        Parameters:
        - wm: the working memory to generate code from
        - recall_res: the recall result from the long term memory

        Preconditions:
        - there exists an item in wm tagged with 'completed_code' after at most 100 recalls.
        """
        if recall_res is None:
            if self._cached_recall_res is None:
                recall_res = self.recall(wm)
            else:
                recall_res = self._cached_recall_res
        n_recall_items = self.n_recall_items
        for i in range(3):
            wm.update_by_recall_res(recall_res, to_tick=True)
            if len(wm.extract_tag_contents("code_suggestion")) > 0:
                break
            else:
                print("No code suggestion found. Recall more agents.")
                n_recall_items += 2
                recall_res = self.recall(wm, n_recall_items)

        agent_res = self.codegen_agent.run_agent(wm)
        recall_res = AgentGroupResult([agent_res])
        wm.update_by_recall_res(recall_res, to_tick=False)
        code = wm.extract_tag_contents("attempted_code")
        if len(code) > 0:
            return code[0]


class CodegenAgent(RetrievableAgent):
    """
    Generate the code based on the working memory
    Will put the generated code in the working memory
    """

    def __init__(self):
        super().__init__("CodegenAgent")

    def get_score(self, w_memory: WorkingMemory):
        if not w_memory.has_tag("code_suggestion"):
            return -1.0
        return 1.0

    def run_agent(self, w_memory: WorkingMemory) -> AgentResult:
        instruction = w_memory.extract_tag_contents("instruction")[0]
        available_variables = w_memory.extract_tag_contents("available_variables")
        if len(available_variables) == 0:
            available_variables = "There is no available variables"
        else:
            available_variables = available_variables[0]
        code_suggestions = w_memory.extract_tag_contents("code_suggestion")
        code_suggestions = [
            "<code_suggestion>\n" + suggestion + "\n</code_suggestion>\n"
            for suggestion in
            code_suggestions]
        code_suggestions = "".join(code_suggestions)
        prompt = f"""
Your task is to generate new code for the context described below.        
<context>
<available_variables>
{available_variables}
</available_variables>
<instruction> 
{instruction}
</instruction>
You must use the following code_suggestions to generate the code:
{code_suggestions}
</context>
<requirements>
You are required to adapt code from one of <code_suggestion> that can be used to implement the <instruction>.
- The adapted code should absolutely just be what should implement <instruction>. 
- Some of the <code_suggestion> might be misleading. But you must pick the most relevant one.
You should first output an analysis of which code suggestion should be used to implement <instruction>.
Then, wrapped by ```python and ```, output the new code that can implement <instruction>. The last line of the generated code must be in the format: `experiment_<ExperimentName> = <ExperimentName>(argument1,argument2, ...)`. The code must be executable. No placeholders are allowed. No import statements are allowed.
</requirements>
        """
        chat = Chat(prompt)
        code = chat.complete(parse="quotes", expensive=True)
        agent_res = AgentResult(self, True)

        if code.startswith("```"):
            code = Parse.quotes(code)

        code_item = CodeWMemoryItem(code, tag="attempted_code")
        agent_res.add_new_wm_item(code_item)
        agent_res.tags_to_remove = ["attempted_code",
                                    "code_suggestion"]  # remove the old attempted code
        return agent_res


def filter_code_suggestion(w_memory: WorkingMemory, instruction):
    code_suggestions = w_memory.extract_tag_contents("code_suggestion")
    code_suggestions = ["<code_suggestion>\n" + suggestion + "\n</code_suggestion>\n"
                        for suggestion in
                        code_suggestions]
    code_suggestions = "".join(code_suggestions)
    prompt = f"""
    You are required to analyze which code suggestion is most suitable for implementing the instruction.
    <instruction>
    {instruction}
    </instruction>
    <code_suggestions>
    {code_suggestions}
    </code_suggestions>
    <requirements>
    - The most suitable code suggestion should be the one that can be used to implement the instruction.
    - If multiple code suggestions are suitable, you should pick the suggestion that matches most to the instruction.
    - You must only analyze the code suggestions and not generate any new code by yourself.
    </requirements>
    <output>
    You are required to output a JSON dict with the following keys:
    - "analysis1" (string): An analysis of which code suggestion is suitable for implementing the instruction.
    - "analysis2" (string): An analysis of which code suggestion is most suitable for implementing the instruction. You may rethink the analysis1 and correct it if necessary.
    - "suggestion" (string): a copy of a single code suggestion that is most suitable for implementing the instruction. 
    </output>
    """
    chat = Chat(prompt, dedent=True)
    res = chat.complete(parse="dict", expensive=True)
    return res["suggestion"]


class CodegenAgentUltraFiltered(CodegenAgent):


    def run_agent(self, w_memory: WorkingMemory) -> AgentResult:
        instruction = w_memory.extract_tag_contents("instruction")[0]
        available_variables = w_memory.extract_tag_contents("available_variables")
        if len(available_variables) == 0:
            available_variables = "There is no available variables"
        else:
            available_variables = available_variables[0]
        code_suggestion_filtered = filter_code_suggestion(w_memory, instruction)
        prompt = f"""
Your task is to generate code for the instruction based on the context described below.        
<context>
<available_variables>
{available_variables}
</available_variables>
<instruction> 
{instruction}
</instruction>
<code_suggestion>
You must use the following code_suggestion to generate the code:
{code_suggestion_filtered}
</code_suggestion>
</context>
<requirements>
You are required to adopt code from <code_suggestion> that can be used to implement <instruction>.
- If the code from the suggestion is not executable due to Python syntax error, you should rewrite it to make it executable.
- You should first output an analysis of the code suggestion.
- Then, wrapped by ```python and ```, output the new code that can implement the <instruction>. The last line of the generated code must be in the format: `experiment_<ExperimentName> = <ExperimentName>(argument1,argument2, ...)`.
 - The code must be executable. No placeholders are allowed. No import statements are allowed.
</requirements>
        """
        chat = Chat(prompt)
        code = chat.complete(parse="quotes", expensive=True)
        agent_res = AgentResult(self, True)

        if code.startswith("```"):
            code = Parse.quotes(code)

        code_item = CodeWMemoryItem(code, tag="attempted_code")
        agent_res.add_new_wm_item(code_item)
        agent_res.tags_to_remove = ["attempted_code",
                                   "code_suggestion"]  # remove the old attempted code
        return agent_res


def get_code_from_wm(wm: WorkingMemory) -> str:
    """
    Extracts code from the working memory that contains code items.

    Parameters:
        wm (WorkingMemory): The working memory to extract code from.

    Returns:
        str: The extracted code.
    """
    code = ""
    for item in wm._items:
        if isinstance(item, CodeWMemoryItem):
            code = item.content
            break
    return code


def get_codegen_wm(description: str, var_table: VariableTable) -> WorkingMemory:
    """
    Prepares working memory for code generation based on a description and variable table.

    Parameters:
        description (str): The description of the code to generate.
        var_table (VariableTable): The variable table to use in the code generation.
        hint (str): The hint to display in the working memory.
    Returns:
        WorkingMemory: The working memory prepared for code generation.
    """

    wm = WorkingMemory()
    if not var_table.is_empty():
        var_table_in_prompt = get_var_table_prompt(var_table)
        var_table_item = WMemoryItem(var_table_in_prompt, "available_variables")
        var_table_item.set_no_prompt()
        var_table_item.attrs["_table_obj"] = var_table
        wm.add_item(var_table_item.set_no_stimuli())
    wm.add_item(WMemoryItem(description, tag="instruction"))
    return wm


def get_var_table_prompt(var_table: VariableTable) -> str:
    lines = []
    for name, obj in var_table.variable_objs.items():
        if isinstance(obj, (int, float, str, bool)):
            lines.append(f"- VariableName: {name}: {obj}")
        else:
            lines.append(f"- VariableName: {name}: {type(obj)}")
    return "\n".join(lines)

def prepare_translation_agents(module, document_folder: str = None, n_agents_to_call: int = 3, lambda_ai_exp_class=None, add_class_to_var_table=None):
    """
    Initialize the translation agent for the experiments
    """

    document_paths = []
    if document_folder is not None:
        for file in os.listdir(document_folder):
            if file.endswith(".md"):
                document_paths.append(os.path.join(document_folder, file))

    agent_group, exp_var_table = build_code_trans_agents(module, document_paths, lambda_ai_exp_class, add_class_to_var_table)

    translation_agents = TranslationAgentGroup()

    for agent in agent_group.agents:
        translation_agents.translation_agents.add_agent(agent)
    translation_agents.n_recall_items = n_agents_to_call

    moduler_var_table = VariableTable()
    moduler_var_table.add_variable("np", np)
    moduler_var_table.add_variable("numpy", np)

    translation_var_table = VariableTable()
    translation_var_table.add_parent_table(exp_var_table)
    translation_var_table.add_parent_table(moduler_var_table)

    return translation_agents, translation_var_table

def init_translation_agents(module, n_agents_to_call: int = 3, document_folder: str = None):
    translation_agents, translation_var_table = prepare_translation_agents(module, document_folder, n_agents_to_call)
    env = TranslationAgentEnv()
    env.translation_agents = translation_agents
    env.translation_var_table = translation_var_table


def build_code_trans_agents(module, document_paths: List[str] = None, is_ai_exp_class: Callable[[Type[Any]], bool] = None, _add_class_to_var_table=None):
    agent_group = AgentGroup()
    var_table = VariableTable()
    if is_ai_exp_class is None:
        is_ai_exp_class = lambda x: issubclass(x, Experiment)
    if _add_class_to_var_table is None:
        _add_class_to_var_table = add_exp_to_var_table

    # Load the module root and scan for experiment classes
    module_root = get_tree_for_module(module)
    classes = []

    for node in module_root.iter_subtree_with_dfs():
        if get_type(node) == "class":
            class_obj = get_obj(node)
            if not is_ai_exp_class(class_obj):
                continue
            classes.append(class_obj)

    def _add_exp_to_agent_group(exp_cls: Type[Any]):
        add_exp_to_agent_group(agent_group, exp_cls)
        _add_class_to_var_table(var_table, exp_cls)

    p_map(_add_exp_to_agent_group, [cls for cls in classes],
          title="Adding experiment to memory")

    document_paths = document_paths or []
    if len(document_paths) > 0:
        extract_procedures_to_agent_group(document_paths, agent_group, var_table)

    return agent_group, var_table
