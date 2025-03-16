import inspect
from typing import Any, Type, List

from mllm import Chat
from mllm.utils.parser import Parse

from k_agents.agent_group.agent_group import AgentGroup, EmbedAgent, AgentResult
from k_agents.agent_group.w_memory import WorkingMemory
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


def add_exp_to_agent_group(agent_group: AgentGroup,
                           exp_cls: Type[Any]) -> None:
    """
    Add an agent of an experiment class to the agent group and variable table for the experiment class.

    Args:
        agent_group (AgentGroup): The agent group
        exp_cls (Type[Any]): The experiment class to be added to agent_group and var_table.
    """
    agent = ExpCodeTranslationAgent(exp_cls)
    agent_group.add_agent(agent)

def add_exp_to_var_table(var_table: VariableTable, exp_cls: Type[Any]) -> None:
    def run_exp(*args, **kwargs):
        return exp_cls()._run(*args, **kwargs)
    var_table.add_variable(exp_cls.__name__, run_exp, exp_cls.__name__)


class ExpCodeTranslationAgent(EmbedAgent):
    def __init__(self, exp_cls: Type[Any]):
        """
        Initialize an agent for triggering and embedding experiment-based sentences.

        Args:
            exp_cls (Type[Any]): The experiment class to be considered.
        """
        exp_name = exp_cls.__name__
        self.exp_name = exp_name
        self.exp_cls = exp_cls
        if "needing_situation" in exp_cls.__dict__:
            embedding_src = exp_cls.needing_situations
        else:
            # Generating sentences for the agent
            embedding_src = imagine_applications(exp_cls.__name__,
                                                 inspect.getdoc(exp_cls.run))
        triggering_src = [exp_name] + embedding_src
        super().__init__(f"{exp_name} suggestion", triggering_src)

    def run_agent(self, w_memory: WorkingMemory) -> AgentResult:
        """
        Execute the agent using the provided working memory, returning an agentResult.

        Args:
            w_memory (WorkingMemory): The current working memory instance.

        Returns:
            AgentResult: The result of executing the agent, possibly modifying working memory.
        """
        # Create a detailed prompt for the Chat model
        instruction = w_memory.extract_tag_contents("instruction")[0]
        available_variables = w_memory.extract_tag_contents("available_variables")
        if len(available_variables) == 0:
            available_variables = "There is no available variables"
        else:
            available_variables = available_variables[0]
        prompt = f"""
You are trying to call an experiment to implement an instruction in Python. The description of the task is written in the slot.
<experiment>
{self.get_exp_description()}
</experiment>
<instruction>
{instruction}
</instruction>
<available_variables>
{available_variables}
</available_variables>
<requirements>
You should output a JSON dict. The keys should be
- "experiment_name_in_slot" (string): The name of the experiment extracted from the slot.
- "analysis" : The brief analysis of the relation between the experiment and the code_to_complete. You should notice that the code_to_complete might be irrelevant to the experiment. You should be careful not to assume additional information. The experiment should considered irrelevant if it contains extra keywords or irrelevant information.
- "applicable": A boolean whether the experiment you hold is suitable for implementing the task. 
- "code": A code snippet that is helpful for filling the slot. The last line of the snippet must be in the format: `experiment_<name> = {self.exp_cls.__name__}(argument1,argument2, ...)`. No import statements are needed.
- "explanation": A detailed explanation of what the code snippet does based solely on the documentation.
- "suitable": A boolean whether the code snippet matches the task based on the documentation.
</requirements>
"""
        chat = Chat(prompt)
        res = chat.complete(parse="dict", expensive=True)

        if not res["applicable"] or not res["suitable"]:
            agent_res = AgentResult(self, False)
            return agent_res

        agent_res = AgentResult(self, True)

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
        agent_res.add_new_wm_content(code, tag="code_suggestion")

        return agent_res

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

    def get_score(self, w_memory: WorkingMemory):
        stimuli = w_memory.stimuli
        for stim in stimuli:
            if self.exp_name in stim:
                return 2.0
        return EmbedAgent.get_score(self, w_memory)
