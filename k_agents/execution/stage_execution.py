from typing import Optional

from k_agents.experiment.experiment import Experiment
from k_agents.io_interface import display_impl
from k_agents.variable_table import VariableTable

import mllm

class Stage:
    """Represents a stage in an experimental workflow."""

    def __init__(self, label: str, title: str, description: str, next_stage_guide: str):
        self.title = title  # Title of the stage
        self.label = label  # Unique identifier for the stage
        self.description = description  # Description of what happens in this stage
        self.next_stage_guide = next_stage_guide  # Guidance for transitioning to the next stage
        self.var_table: VariableTable = VariableTable()  # Variable table specific to this stage, initialized later
        self.n_failed = 0  # Number of times the stage has been executed
        self.n_executed = 0  # Number of times the stage has been executed successfully
        self.n_success = 0  # Number of times the stage has been executed successfully

    def to_dict(self) -> dict:
        """Converts the stage to a dictionary."""
        return {
            "Label": self.label,
            "Title": self.title,
            "ExperimentDescription": self.description,
            "Next": self.next_stage_guide,
            "Variables": self.var_table.get_prompt(no_doc=True)
        }

    def to_xml(self) -> str:
        """Converts the stage to an XML string."""
        stage_dict = self.to_dict()
        xml_str = "<Stage>\n"
        for key, value in stage_dict.items():
            xml_str += f"  <{key}>{value}</{key}>\n"
        xml_str += "</Stage>"
        return xml_str

    def display(self):
        """
        Display information about the stage in a jupyter notebook.
        First converts the stage to a dictionary into a markdown format.
        then display it using IPython
        """
        from IPython.display import Markdown

        stage_markdown = f"""##{self.title}
        **Label**: {self.label}
        **Description**: {self.description}
        **Next Stage Guide**: {self.next_stage_guide}
        """
        display_impl(Markdown(stage_markdown))


def get_exp_from_var_table(var_table: VariableTable) -> Optional[Experiment]:
    """Searches the variable table for an Experiment object."""
    for name, obj in var_table.variable_objs.items():
        if isinstance(obj, Experiment):
            return obj


def check_if_needed_to_break_down(description: str):
    """
    Check if the stage described by the description needs to be broken down into multiple steps.

    Parameters:
        description (str): The description of the code to generate.
    """

    # Check if we need to split the stage described by the description into multiple steps
    prompt = f"""
    Evaluate whether a stage description in an experimental process should be executed as a single step or needs to be divided into multiple steps. Consider the following criteria for your evaluation:

    - Multiple Experiments: If the stage includes multiple distinct experiments, each requiring unique parameters, divide it into multiple steps.
    - Repeated Experiments with Specified Parameters: If the stage involves repeating one experiment multiple times with different parameters that are explicitly specified, divide it into multiple steps.
    - Repeated Experiments without Specified Parameters: If the stage involves repeating one experiment multiple times without explicitly specified parameters, treat it as a single step.
    - Parameter Variations Within an Experiment: If a single experiment includes variations in parameters, consider these as part of the internal management of the experiment and do not divide the stage.
    - Single Experiment, No Specified Parameters: If the stage comprises only one experiment with no parameters specified, execute it as a single step.
    - If this is not the first attempt at the stage, do not divide it into multiple steps.
    
    If division into multiple steps is necessary, provide a detailed description of each step, including the parameters for each.

    Stage Description:
    {description}

    Expected return format:
    {{
        "reason": str,
        "single_step": bool,
        "steps": List[dict]
    }}
    """

    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict", dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    return res


