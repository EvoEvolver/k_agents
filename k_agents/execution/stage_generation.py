import json
from pprint import pprint
from typing import List, Dict

import mllm
from mllm import Chat
from mllm.utils import p_map

from k_agents.execution.stage_execution import Stage


def extract_stages(description: str) -> Dict:
    prompt = f"""
<experiment_description>
{description}
</experiment_description>
<objective>
Your objective is to decompose the experiment description into standalone instruction.
Each instruction should include an experiment. 
The instruction should make a minimal modification to the original description.
You should not do any inference or interpretation of the description. 
You are encouraged to copy the description as is.
You should output as few instructions as possible. You must not expand the instructions.
The instructions must not contain any information about what to do next after the instruction, such as a change of parameter and go to fail.
</requirements>
<example>
For example, if a piece of description is:
"Run experiment A with frequency=10. If failed, retry 3 times."
You should change it into:
"Run experiment A with frequency=10."
<output_format>
You are required to output a JSON dict with a single key "instructions", which contains a list of instructions. Each instruction should be represented as a string.
</output_format>
"""

    completed_prompt = prompt

    chat = Chat(completed_prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict", dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    stages = {}
    for i, stage in enumerate(res["instructions"]):
        stages[f"Stage{i+1}"] = stage

    return stages


def extract_parameters(description) -> dict:
    prompt = f"""
<objective>
Your objective is to extract the parameters from a given description of an experiment.
</objective>
<description>
{json.dumps(description, indent=1)}
</description>
<requirement>
You are required to extract parameters of the experiment from the given description.
However, non-experimental parameters, such as the number of retries, should not be extracted.
You are required to put the unit of the parameter in its name instead of the value.
For example:
"Do experiment A with frequency offset 10MHz and amplitude=`amplitude` and duration=10us. If failed, retry 3 times."
should be extracted as:
"Do experiment A with frequency_offset=`frequency_offset` and amplitude=`amplitude` and duration. If failed, retry 3 times."
and
{{
    "frequency_offset_in_MHz": 10,
    "amplitude": None,
    "duration_in_us": None
}}
</requirement>
<output_format>
You are required to output a JSON dict with the following keys:
"new_description" (dict): The description with the parameters replaced by placeholders. The dict must in the same format as the input description.
"parameters" (dict): The extracted parameters in a dict. the key of the dict must be a valid variable name in Python.
"""

    completed_prompt = prompt

    chat = Chat(completed_prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict", dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    return res

def attach_next_stage_guide(stages: Dict, description: str) -> Dict:
    prompt = f"""
You are required to attach the next stage guide to each stage in the following given list of stages.
<stages>
{json.dumps(stages, indent=1)}
</stages>
<experiment_description>
{description}
</experiment_description>
<requirement>
- You are required to attach the next stage guide to each stage in the given list of stages.
- The next stage guide is the instruction of what to do next after the stage. 
- If may include conditions for the transition to the next stage.
- By default, after each stage, go to the next stage in the list.
- Remember that there exist two special stages: Failed and Complete.
- Especially, you must translate the transition rule into a form using goto StageX.
- You should add a clear condition on the goto statements.
- If the experiment_description does not contain the next stage guide, just use the default rule: Fail if the stage fails, go to next stage if the stage completes.
</requirement>
<output_format>
You are required to output a JSON dict containing a list of stages. The number of stages should be the same as the input list.
{{
   "stage_analysis": str,
   "Stage1": {{
   "instruction": str,
   "original_next_stage_guide": str
   "next_stage_guide_with_goto": str
   }},
    "...": {{
    }}
}}
</output_format>
"""

    completed_prompt = prompt

    chat = Chat(completed_prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict", dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=False)
    del res["stage_analysis"]
    stage_dict = {key: {"instruction": value["instruction"], "next_stage_guide": value["next_stage_guide_with_goto"]}
                  for key, value in res.items()}

    return stage_dict


def generate_stages(description):
    stage_titles = []
    raw_stages = extract_stages(description)
    for i, stage_dict in enumerate(raw_stages.values()):
        stage_titles.append(f"Stage{i+1}")
    raw_stages["Complete"] = "experiment is completed."
    raw_stages["Failed"] = "experiment is fail"
    raw_stages = attach_next_stage_guide(raw_stages, description)
    del raw_stages["Complete"]
    del raw_stages["Failed"]
    raw_stages_with_parameters = []
    parameters_list = []
    for s, res in p_map(extract_parameters, raw_stages.values()):
        raw_stages_with_parameters.append(res["new_description"])
        parameters = res["parameters"]
        number_parameters = {}
        for key, value in parameters.items():
            if isinstance(value, (int, float, bool)):
                number_parameters[key] = value
        parameters_list.append(number_parameters)
    stages = []
    for i, raw_stage in enumerate(raw_stages_with_parameters):
        stage = Stage(label=f"Stage{i+1}", title=stage_titles[i],
                      description=raw_stage["instruction"],
                      next_stage_guide=raw_stage["next_stage_guide"])
        stages.append(stage)
        stage.var_table.update_by_dict(parameters_list[i])
    return stages


if __name__ == '__main__':
    description_rabi = '''
- Iterative Two-qubit Amplitude test at frequency=4800 on `duts`
    '''

    description = description_rabi
    generate_stages(description)


def stages_to_html(stages_list: List[Stage]):
    stages_dict = {stage.label: stage.to_dict() for stage in stages_list}

    html_content = '<div style="font-family: Arial, sans-serif;">'

    # Loop through each stage in the dictionary
    for stage_key, stage_info in stages_dict.items():
        # Skip "Complete" and "Failed" stages
        if stage_key in ["Complete", "Failed"]:
            continue

        if stage_info['Variables'].strip() == "":
            variables = ""
        else:
            variables = f"<p><strong>Variables:</strong> <pre>{stage_info['Variables']}</pre></p>"

        # Adding HTML content for each stage
        html_content += f'''
            <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                <h3>{stage_info['Title']}</h3>
                <p><strong>Description:</strong> {stage_info['ExperimentDescription']}</p>
                <p><strong>Next Steps:</strong> {stage_info['Next']}</p>
                {variables}
            </div>
        '''

    html_content += '</div>'
    return html_content


def remove_unused_stages_and_update_next(stage_info_list: List[dict]) -> List[dict]:
    """
    Remove unused stages and update the next stage information.
    """

    prompt = f"""
You have created a list of stages for an experiment. Your task is to modify this list based on specific criteria:

- Identify and remove any stages that are marked with 'contains_experiment=False'. Assume these stages are successful by default.
- For the remaining stages, update the rule for transitioning to the next stage based on the results of the experiment.
- Keep the rest information of each stage unchanged, return in the same format as the input.
- Keep the 'Complete' and 'Failed' stages as the final stages of the experiment.

This process ensures that the list reflects only the stages actively involved in the experiment and adjusts the workflow according to experimental outcomes.

<stages>
{json.JSONEncoder().encode(stage_info_list)}
</stages>

Return format:

{{
"stages": List[dict]
}} 
"""

    chat = mllm.Chat(prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict")
    updated_stage_info = chat.complete(parse="dict", expensive=True, cache=True)
    return updated_stage_info["stages"]


def find_the_stage_label_based_on_description(stages: List[Stage], description: str):
    """
    Find the stage label based on the description.

    Parameters:
        stages (List[Stage]): The list of stages.
        description (str): The description to search for.

    return (Stage): The stage.
    """

    stages_info_lines = []
    for stage in stages:
        stages_info_lines.append(f"- {stage.label}")
    stages_info_lines.append("- Complete")
    stages_info_lines.append("- Failed")
    stages_info = "\n".join(stages_info_lines)

    prompt = f"""
    You have a list of stages for an experiment. Your task is to find the stage label based on the description provided.

    <description>
    {description}
    </description>
    
    Available stages:
    <stages>
    {stages_info}
    </stages>

    Return format:
        {{
        “analysis”: str,
        "stage_label": str
        }}
    """

    chat = mllm.Chat(prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict", dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    for stage in stages:
        if res['stage_label'] in stage.label or stage.label in res['stage_label']:
            return stage
