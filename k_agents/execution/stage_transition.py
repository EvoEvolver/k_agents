from __future__ import annotations

import json
from typing import TYPE_CHECKING

import mllm

if TYPE_CHECKING:
    from .stage_execution import Stage


def generate_parameter_patch(stage_jumped_to: Stage, inspection_result) -> dict:

    variables_in_prompt = {}
    for var_name, var_obj in stage_jumped_to.var_table.variable_objs.items():
        # if the var_obj is a number, str, or bool, we can just use it as is
        if isinstance(var_obj, (int, float, str, bool)):
            variables_in_prompt[var_name] = var_obj


    prompt = f"""
    Based on the information provided, you have transitioned to a new stage, identified as {stage_jumped_to.label}.
    
    The current description of this stage is:
    <description>
    {stage_jumped_to.description}
    <description>
    The current stage is using the following parameters:
    <parameters>
    {json.dumps(variables_in_prompt, indent=1)}
    </parameters>
    <report>
    {inspection_result}
    </report>
    <requirements>
    Using the experiment report provided, decide whether you need to updated parameters for this stage. 
    If so, provide how to update the parameters.
    Response in JSON with the following keys:
    "analysis" (string): an analysis about whether the parameters need to be updated and how to update them if needed. The parameter update is needed only when it is mentioned in the report.
    "to_update" (bool): whether the parameters need to be updated.
    "parameter_patch" (dict):  a dict that describes how to update the parameters. The keys should be the variable names and the values should be the new values. If a parameter is not updated, it should not be included in the dict.
    </requirements>
    """

    chat = mllm.Chat(prompt, dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    to_update = res["to_update"]
    if not to_update:
        return {}
    parameter_patch = res["parameter_patch"]
    return parameter_patch


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
    if "success" in experiment_result:
        if experiment_result['success']:
            current_stage.n_failed = 0
            current_stage.n_success += 1
        else:
            current_stage.n_failed += 1
    current_stage.n_executed += 1

    result_prompt = ""

    for key, value in experiment_result.items():
        result_prompt += f"{key}: {value}\n\n"

    prompt = f"""
    You are analyzing experiment result from current stage and use the rule of transition to determine the next stage of the experiment.
    
    <current_stage>
    {current_stage.label}:{current_stage.description}
    """

    prompt += f"""
    The current stage has been executed {current_stage.n_executed} times.
    There are {current_stage.n_failed} failed attempts and {current_stage.n_success} successful attempts.
    """

    prompt += f"""
    </current_stage>
   
    Here are the results from the experiments. Note that results must be consistent to indicate the validity. 
    Otherwise they are both invalid.
    <experiment_reports>
    {result_prompt}
    </experiment_reports>
    
    <rule_of_transition>
    {rules}
    </rule_of_transition>
    
    <requirements>
    Return your decision in JSON format With the following keys:
    "analysis" (string): an analysis of the results and the rule of transition to determine the next stage.
    "next" (string): the name of the next stage.
    </requirements>
    """

    chat = mllm.Chat(prompt, dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    return res


