from __future__ import annotations
from typing import TYPE_CHECKING

import mllm

if TYPE_CHECKING:
    from .stage_execution import Stage


def generate_new_stage_description(stage_jumped_to: Stage) -> str:
    """
    Generate a new stage of the stage jumped to and additional information provided.

    Parameters:
        stage_jumped_to (Stage): The stage that was jumped to.

    Returns:
        new_description (str): The new description of the stage.
    """

    prompt = f"""
    Based on the information provided, you have transitioned to a new stage, identified as {stage_jumped_to.label}.
    
    The current description of this stage is:
    <description>
    {stage_jumped_to.description}
    <description>
    
    Using the details provided, write an updated description for this stage. Furthermore, if there are instructions in the
    additional information to adjust certain parameters, select specific values for each parameter as requested and justify
    these choices based on the analysis provided. Include your analysis only in the analysis field and aim for conciseness
    and clarity in your revised description. Do not include the objective into the description.
    
    Example of the description:
    "Conduct the <experiment name> with parameters <parameter list for experiment>."
    
    Follow the example exactly and do not include any additional information in the description.
    
    Response in JSON with the following keys:
    "analysis" (string):  your thought process for updating the stage description.
    "new_description" (string):  the updated description of the stage here.
    """

    chat = mllm.Chat(prompt, dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    new_description = res["new_description"]

    return new_description


def get_next_stage_label(current_stage: Stage, experiment_result: dict[str, str]) -> dict[str, str]:
    """
    Get the next stage label based on the current stage and the experiment object.

    Parameters:
        current_stage (Stage): The current stage.
        experiment_result (dict[str,str]): The experiment results.

    Returns:
        next_stage_label (dict[str,str]): The next stage label and the additional information for executing
        the next stage.
    """
    rules = current_stage.next_stage_guide

    if experiment_result['success']:
        current_stage.n_failed = 0
    else:
        current_stage.n_failed += 1

    result_prompt = ""

    for key, value in experiment_result.items():
        result_prompt += f"{key}: {value}\n\n"

    prompt = f"""
    You are operating a state machine and the current stage has produced some results. 
    You must analyze these results and use the rule of transition to determine the next stage of the machine.
    
    <current_stage>
    {current_stage.label}:{current_stage.description}
    """
    if current_stage.n_failed > 0:
        prompt += f"""The current stage has failed {current_stage.n_failed} times."""

    prompt += f"""
    </current_stage>
    
    <rule_of_transition>
    {rules}
    </rule_of_transition>
    
    Here are the results from the experiments. Note that results must be consistent to indicate the validity. 
    Otherwise they are both invalid.
    <experiment_reports>
    {result_prompt}
    </experiment_reports>
    
    <requirements>
    Return your decision in JSON format With the following keys:
    "analysis" (string): an analysis of the results and the rule of transition to determine the next stage.
    "next" (string): the name of the next stage.
    </requirements>
    """

    chat = mllm.Chat(prompt, dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    return res


