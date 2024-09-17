import inspect
import os
from typing import Any, Tuple, Type, List

from fibers.data_loader.module_to_tree import get_tree_for_module
from fibers.tree.node_attr.code import get_type, get_obj
from mllm import Chat
from mllm.utils import parallel_map
from mllm.utils.parser import Parse


from k_agents.ideanet.lt_memory import IdeaResult, LongTermMemory, EmbedIdea
from k_agents.ideanet.w_memory import WorkingMemory
from k_agents.variable_table import VariableTable


def imagine_applications(exp_name, exp_docs) -> List[str]:
    """
    Generate a list of imperative sentences based on the documentation of an experiment class method.

    Args:
        exp_cls (Type[Any]): The experiment class.

    Returns:
        List[str]: A list of imperative sentences derived from the experiment's documentation.
    """
    # Retrieve the docstring for the `run` method of the experiment class
    doc_string = exp_docs
    # Construct the prompt for the Chat model
    prompt = f"""
You are trying to produce imperative sentences that would invoke the execution of experiment `{exp_name}` based on its documentation.
<docstring>
{doc_string}
</docstring>
<example>
Here are a few of examples of imperative sentences:
- Run the calibration experiment with duts=duts and start=0.0
- Calibrate `duts` 
- Please execute the CrossAllXYDragMultiSingleQubitMultilevel experiment
- Do the AllXY drag experiment.
</example>
<instruction>
You should output a JSON dict. The keys should be string of indices of the sentences and the values should be the sentences. 
Each sentence should be complete and independent. Name of the experiment should be transformed to natural language and be mentioned.
The sentences should be imperative and should be based on the documentation.
You should output 4 sentences.
</instruction>
"""
    # Instantiate a Chat model and get responses
    chat = Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    # Extract the values from the response dictionary
    values = list(res.values())
    return values


class LeeQExpCodeIdea(EmbedIdea):
    def __init__(self, exp_cls: Type[Any]):
        """
        Initialize an idea for triggering and embedding experiment-based sentences.

        Args:
            exp_cls (Type[Any]): The experiment class to be considered.
        """
        exp_name = exp_cls.__name__
        self.exp_name = exp_name
        self.exp_cls = exp_cls
        if "needing_situation" in exp_cls.__dict__:
            embedding_src = exp_cls.needing_situations
        else:
            # Generating sentences for the idea
            embedding_src = imagine_applications(exp_cls.__name__, inspect.getdoc(exp_cls.run))
        triggering_src = [exp_name] + embedding_src
        super().__init__(f"{exp_name} suggestion", triggering_src)

    def run_idea(self, w_memory: WorkingMemory) -> IdeaResult:
        """
        Execute the idea using the provided working memory, returning an IdeaResult.

        Args:
            w_memory (WorkingMemory): The current working memory instance.

        Returns:
            IdeaResult: The result of executing the idea, possibly modifying working memory.
        """
        # Create a detailed prompt for the Chat model
        instruction = w_memory.extract_tag_contents("instruction")[0]
        available_variables = w_memory.extract_tag_contents("available_variables")
        if len(available_variables) == 0:
            available_variables = "There is no available variables"
        else:
            available_variables = available_variables[0]
        prompt = f"""
You are trying to call a experiment to fill the code_to_complete in Python. The description of the task is written in the slot.
<experiment>{self.get_exp_description()}
</experiment>
<code_to_complete>
# [slot: {instruction}]
</code_to_complete>
<available_variables>
{available_variables}
</available_variables>
<requirements>
You should output a JSON dict. The keys should be
- "evidence": Evidence that indicates the experiment is relevant to the task or not.
- "analysis" : The brief analysis of the relation between the experiment based on the evidence. You should notice that the code_to_complete might be irrelevant to the experiment. You should be careful not assume additional information.
- "applicable": A boolean whether the experiment you hold is suitable for implementing the task. 
- "code": A code snippet that is helpful for filling the slot. The last line of the snippet must be in the format: `experiment_<name> = {self.exp_cls.__name__}(argument1,argument2, ...)`.
- "explanation": A detailed explanation of what the code snippet does based solely on the documentation.
- "suitable": A boolean whether the code snippet matches the task based on the documentation.
</requirements>
"""
        chat = Chat(prompt)
        res = chat.complete(parse="dict", expensive=True)

        if not res["applicable"] or not res["suitable"]:
            idea_res = IdeaResult(self, False)
            return idea_res

        idea_res = IdeaResult(self, True)

        one_line_reason = res["explanation"].replace('\n', ' ')

        suggestion = res["code"]
        if suggestion.startswith("```"):
            suggestion = Parse.quotes(suggestion)

        signature = inspect.signature(self.exp_cls.run)
        # remove the first argument
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        code = f"""
# {one_line_reason}
# Suggested code:
{suggestion}
"""
        idea_res.add_new_wm_content(code, tag="code_suggestion")

        return idea_res

    def get_exp_description(self):
        signature = inspect.signature(self.exp_cls.run)
        # remove the first argument
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])

        return f"""
Experiment signature:
<signature> 
{self.exp_name}{signature}
</signature>
<document>
{inspect.getdoc(self.exp_cls.run)} 
<document>
"""


def add_leeq_exp_to_ltm(lt_memory: LongTermMemory, var_table: VariableTable, exp_cls: Type[Any]) -> None:
    """
    Add an experiment class to the long term memory and variable table for leeq.

    Args:
        lt_memory (LongTermMemory): The long term memory for leeq.
        var_table (VariableTable): The variable table for leeq.
        exp_cls (Type[Any]): The experiment class to be added.
    """
    idea = LeeQExpCodeIdea(exp_cls)
    lt_memory.add_idea(idea)
    var_table.add_variable(exp_cls.__name__, exp_cls, exp_cls.__name__)


def build_leeq_code_ltm(add_document_procedures=True) -> Tuple[LongTermMemory, VariableTable]:
    """
    Build the idea base for leeq. It scans built-in experiments and creates ideas for them.

    Returns:
        Tuple[LongTermMemory, VariableTable]: The long term memory for leeq and the loaded variable table.
    """
    from leeq.experiments import builtin

    lt_memory = LongTermMemory()
    var_table = VariableTable()

    # Load the module root and scan for experiment classes
    module_root = get_tree_for_module(builtin)
    classes = []
    from leeq import Experiment
    for node in module_root.iter_subtree_with_dfs():
        if get_type(node) == "class":
            class_obj = get_obj(node)
            if not issubclass(class_obj, Experiment):
                continue
            elif not class_obj.is_ai_compatible():
                continue
            classes.append(class_obj)

    # Load the AI automated experiment class for nested execution.
    from k_agents.experiment.automation import AutoRun
    # classes.append(FullyAutomatedExperiment)
    # classes.append(AIInstructionExperiment)
    # classes.append(AIRun)
    # classes.append(AutoRun)

    var_table.add_variable('AutoRun', AutoRun, None)

    def _add_leeq_exp_to_ltm(exp_cls: Type[Any]):
        add_leeq_exp_to_ltm(lt_memory, var_table, exp_cls)

    for i, idea in parallel_map(_add_leeq_exp_to_ltm, [cls for cls in classes], title="Adding experiment to memory"):
        ...

    from leeq import experiments as exp
    from k_agents.indexer.procedure_indexer import extract_procedures_to_lt_memory
    root = os.path.dirname(exp.__file__)
    if add_document_procedures:
        extract_procedures_to_lt_memory(root + "/procedures/calibration.md", lt_memory)
    return lt_memory, var_table
