import inspect
from typing import Any, Type, List

from mllm import Chat
from mllm.utils.parser import Parse

from k_agents.memory.lt_memory import LongTermMemory, EmbedIdea, IdeaResult
from k_agents.memory.w_memory import WorkingMemory
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
You are trying to produce imperative sentences that would invoke the execution of action `{exp_name}` based on its documentation.
<docstring>
{doc_string}
</docstring>
<example>
Here are a few of examples of imperative sentences:
- Run the calibration experiment with duts=`duts` and start=0.0
- Carry out a calibration on `duts` 
- Please execute the Ramsey experiment
- Do the Drag experiment.
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


def add_exp_to_ltm(lt_memory: LongTermMemory, var_table: VariableTable,
                   exp_cls: Type[Any]) -> None:
    """
    Add an experiment class to the long term memory and variable table for the experiment class.

    Args:
        lt_memory (LongTermMemory): The long term memory .
        var_table (VariableTable): The variable table.
        exp_cls (Type[Any]): The experiment class to be added to lt_memory and var_table.
    """
    idea = ExpCodeTranslationAgent(exp_cls)
    lt_memory.add_idea(idea)
    var_table.add_variable(exp_cls.__name__, exp_cls, exp_cls.__name__)


class ExpCodeTranslationAgent(EmbedIdea):
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
            embedding_src = imagine_applications(exp_cls.__name__,
                                                 inspect.getdoc(exp_cls.run))
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
<experiment>
{self.get_exp_description()}
</experiment>
<code_to_complete>
# [slot: {instruction}]
</code_to_complete>
<available_variables>
{available_variables}
</available_variables>
<requirements>
You should output a JSON dict. The keys should be
- "experiment_name_in_slot" (string): The name of the experiment extracted from the slot.
- "analysis" : The brief analysis of the relation between the experiment. You should notice that the code_to_complete might be irrelevant to the experiment. You should be careful not assume additional information. The experiment should considered irrelevant if it contains extra keywords or irrelevant information.
- "applicable": A boolean whether the experiment you hold is suitable for implementing the task. 
- "code": A code snippet that is helpful for filling the slot. The last line of the snippet must be in the format: `experiment_<name> = {self.exp_cls.__name__}(argument1,argument2, ...)`. No import statements are needed.
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