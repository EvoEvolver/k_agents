from typing import List

from mllm.utils import parallel_map

from k_agents.execution.stage_execution import Stage
import json
import mllm

def stages_to_html(stages_list):
    stages_dict = {stage.label: stage.to_dict() for stage in stages_list}

    html_content = '<div style="font-family: Arial, sans-serif;">'

    # Loop through each stage in the dictionary
    for stage_key, stage_info in stages_dict.items():
        # Skip "Complete" and "Fail" stages
        if stage_key in ["Complete", "Fail"]:
            continue

        # Adding HTML content for each stage
        html_content += f'''
            <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                <h3>{stage_info['Title']}</h3>
                <p><strong>Description:</strong> {stage_info['ExperimentDescription']}</p>
                <p><strong>Next Steps:</strong> {stage_info['Next']}</p>
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
- Keep the 'Complete' and 'Fail' stages as the final stages of the experiment.

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


def refine_stage_description(res: dict) -> dict:
    """
    Refine the stage description based on the response from the AI.

    Parameters:
        res (dict): The response from the AI.
    """

    prompt = f"""
Please refine the description generated from the input with the following condition. Note that the description
may only reflect part of the input.

<input prompt>
{res["Reference"]}
</input prompt>

<generated title>
{res['Title']}
</generated title>

<generated description>
{res['ExperimentDescription']}
</generated description>

- If the generated description contains information not related to the input, remove it. 
- If the generated description contains objectives or goals, remove them.
- Quote the the parameters and the values in the format of `"<parameter name>=<parameter value>"` if the actual values are present in the description. 
    The values should be the actual values, not placeholders. 
- Only modify the parts described above, keep the rest of the description as is.
- If this stage only contains data and result analysis and interpretation without carrying out any experiment,
  please set the <contains_experiment> to False. Otherwise set it to True.
- If the generated description contains information that is not present in the reference, for example the details
   how to implement each stages but they are not specifically described in the reference, remove these information.

Follow the following example format exactly and strictly and do not include any additional information
 in the description. Do not include any information that is not presented in the description, such as the details
 in how to implement each steps based on your knowledge.

Example output when no parameters and their values are present:
<example 1>
{{
    "analysis":"<Describe your thought process for updating the stage description.>",
    "description":"Conduct the <experiment name>.",
    "contains_experiment": <true/false>
}}
</example 1>

Example output when parameters and their values are present:
 <example 2>
{{
    "analysis":"<Describe your thought process for updating the stage description.>",
    "description":"Conduct the <experiment name> with parameters <parameter list for experiment>.",
    "contains_experiment": <true/false>
}}
</example 2>


"""

    chat = mllm.Chat(prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict")
    updated_res = chat.complete(parse="dict", expensive=True, cache=True)

    new_res = {
        "Title": res["Title"],
        "ExperimentDescription": updated_res["description"],
        "Next": res["Next"],
        'contains_experiment': updated_res["contains_experiment"]
    }

    return new_res


def _get_stage_from_agent_response(stage_info: tuple) -> dict:
    """
    Get a stage object from the response of the AI agent.

    Parameters:
        stage_info (tuple): The tuple containing the stage name and content.

    Returns:
        dict: The stage object.
    """

    stage_name, stage_content = stage_info

    if stage_name in ["Complete", "Fail"]:
        refined_content = stage_content
    else:
        refined_content = refine_stage_description(stage_content)

    stage_content.update(refined_content)

    return stage_content


def get_stages_from_instruction(description: str) -> List[Stage]:
    """
    Get stages from the description of the experiment.

    Parameters:
        description (str): The description of the experiment.

    Returns:
        List[Stage]: The list of stages of the experiment.
    """
    # Note: The same experiment with different parameter choice (very common when you need to refine the parameters) needs to be classified into the same stage. #

    prompt = f"""
<experiment_description>
{description}
</experiment_description>
<requirements>
Create a structured workflow for conducting a series of scientific experiments that require sequential function calls to operate experimental equipment. Each stage of the experiment should represent a distinct operation with explicit instructions and transition rules, ensuring logical and smooth progression. The stages must be concise, self-contained, and clearly defined. Especially, you are required to output a JSON dict with the following elements:
</requirements>
<output>
- Overview: Summarize the requirement of the experiment. Only include the information from the input, do not add any additional information of your own.

- Stages: Divide the experiment into distinct stages, each representing a specific operation. 
Note: Generate as less stages as possible, ideally just one stage, but make sure each stage is distinct and has a clear purpose.
Note: If multiple sets of parameters are used for the same experiment, they should be considered into different stages.
Note: The data and result analysis and interpretation should not be considered as a stage.
Note: Refinement of the parameters should be included in the same stage, not in a separate stage.
Note: Do not include any additional information that is not present in the description, for example the details how to implement each stages.

- ExperimentDescription: Provide a detailed procedural outline for each stage of the experiment. The description should explicitly state the name of the experiment, list all parameters involved, and clearly outline the steps to be taken. This information will be distributed among various team members, who will carry out the tasks. Ensure that each instruction is clear and self-sufficient, enabling team members to execute their respective parts without needing additional context or clarification. Do not include objectives or goals in the description.

- StageTransitions:
    - `Advance`: Always proceed to the next stage when the experiment succeeded.
    - `Retry`: Specify when to repeat a stage with adjustments. Always retry at least 3 times when error occured 
        before revert to the previous stage.
    - `Revert`: Return to the previous stage.
- Reference: Include the original input prompt related to each stage for reference and context.
</output>
<output_format>
Present these instructions and conditions in a JSON format, with each stage as a key detailing the experiment and transition rules. 
The NEXT key must be a string detailing the transition conditions. Do not use "retry", "advance", or "revert", instead describe the stage label directly.
</output_format>""" + """
<output_example>
{
  "Overview": <Summary of the experiment>
  "Stage1": {
    "Title": "Experiment1",
    "ExperimentDescription": "Conduct the <experiment name 1> with parameters <parameter list for experiment 1>.",
    "Next": "Proceed to Stage2 if successful, adjust the parameter based on the results suggestion and retry Stage1 if not. After 3 failures, proceed to Fail."
    "Reference":'<The original input prompt related to this stage>'
  },
  "Stage2": {
    "Title": "Experiment2",
    "ExperimentDescription": "Conduct the <experiment name 2> with parameters <parameter list for experiment 2>.",
    "Next": "Move to Complete if successful, return to Stage2 if inconclusive. After 3 failures, proceed to Fail."
    "Reference":'<The original input prompt related to this stage>'
  },
  "Complete": {
    "Title": "Completion",
    "ExperimentDescription": "Conclude the experiment has succeeded.",
    "Next": "None"
  },
  "Fail": {
    "Title": "Failure",
    "ExperimentDescription": "Conclude the experiment has failed.",
    "Next": "None"
  }
}
</output_example>"""

    completed_prompt = prompt

    chat = mllm.Chat(completed_prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", expensive=True, cache=True)
    stages = []

    overview = res.pop("Overview")

    # Add overview to each dict in res
    for stage_name, stage_content in res.items():
        stage_content['Overview'] = overview
        stage_content['label'] = stage_name

    stages_info = [k[1] for k in
                   sorted(parallel_map(_get_stage_from_agent_response, res.items()),
                          key=lambda x: x[0])]

    # Check if there is any stage marked as contains_experiment=False
    has_stage_need_to_remove = len([stage for stage in stages_info if
                                    stage['label'] not in ['Complete', 'Fail'] and not
                                    stage['contains_experiment']]) > 0

    if has_stage_need_to_remove:
        stages_info = remove_unused_stages_and_update_next(stages_info)

    for stage_info in stages_info:
        stage = Stage(label=stage_info['label'], title=stage_info['Title'],
                      overview=stage_info['Overview'],
                      description=stage_info['ExperimentDescription'],
                      next_stage_guide=stage_info['Next'])
        stages.append(stage)

    # for stage_name, stage_content in res.items():

    #    if stage_name in ["Complete", "Fail"]:
    #        refined_content = stage_content
    #    else:
    #        refined_content = refine_stage_description(stage_content)

    #    stage = Stage(label=stage_name, title=refined_content['Title'],
    #                  overview=overview,
    #                  description=refined_content['ExperimentDescription'],
    #                  next_stage_guide=refined_content['Next'])
    #    stages.append(stage)

    return stages


def find_the_stage_label_based_on_description(stages: List[Stage], description: str):
    """
    Find the stage label based on the description.

    Parameters:
        stages (List[Stage]): The list of stages.
        description (str): The description to search for.

    return (Stage): The stage.
    """

    stages_info = '\n'.join([stage.to_xml() for stage in stages])

    prompt = f"""
    You have a list of stages for an experiment. Your task is to find the stage label based on the description provided.

    <stages>
    {stages_info}
    </stages>

    <description>
    {description}
    </description>

    Return format:
        {{
            "stage_label": str
        }}
    """

    chat = mllm.Chat(prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", expensive=True, cache=True)

    for stage in stages:
        if res['stage_label'] in stage.label:
            return stage


if __name__ == '__main__':
    description_ramsey = '''
# Gate Frequency Calibration
## Background

Ramsey experiment can predict the qubit frequency different to the frequency I am driving it. First I guess a qubit frequency (which already set in the system), and assume the difference is no more than 10 MHz. Therefore I run a Ramsey experiment with frequency offset 10 MHz. Then I wish to do a more accurate calibration by increase the experiment time, and reduce the offset to 1MHz. If this experiment failed and show a value more than 3 MHz its likely that the initial guess is more than 10MHz away from the qubit. Therefore we go back and run experiment at 20MHz offset again. After it succeeded, we do a fine calibration with offset 0.1MHz.

## Steps

- Run Ramsey experiment on the qubit, with frequency offset 10 MHz, stop at 0.3us, step 0.005us.
- Extract the number of periods from the AI analysis texts.
- If observed less than 3 period, double the stop value and step value and try again.
- If observed more than 10 period, half the stop value and step value and try again.
- Run Ramsey experiment on the qubit, with frequency offset 1 MHz, stop at 3us, step 0.05us
- If the second experiment obtained a frequency offset more than 3MHz, go back to the first step, set frequency offset to 20MHz. and try again.
- Run Ramsey experiment on the qubit, with frequency offset 0.1 MHz, stop at 30us, step 0.5us.
'''
    description_rabi = '''

# Gate Amplitude Calibration

## Background

To accurately calibrate the amplitude of the control pulses for our qubit gates, we start with a Rabi oscillation experiment. This experiment helps determine the amplitude required to perform a full rotation on the Bloch sphere. We begin the calibration with a preliminary range of pulse durations starting from 0.01 microseconds up to 0.15 microseconds, incrementing by 0.001 microseconds each step. Successful determination of the Rabi frequency from these measurements will indicate the optimal amplitude setting for the qubit gates.

After successfully calibrating the Rabi frequency, we proceed to Pingpong amplitude calibration using the default parameters. This secondary calibration further refines our control over the qubit by adjusting the amplitudes based on the results from the Rabi experiment, ensuring more precise and reliable gate operations.

## Steps

- Conduct a Rabi experiment to determine the Rabi frequency: Start pulse duration at 0.01 microseconds, step 0.001 microseconds, stop at 0.15 microseconds.
- If observed less than 3 period, double the stop value and step value and try again..
- If observed more than 10 period, half the stop value and step value and try again.
- If the above experiment failed, re-do it and adjust parameters based on visual instructions.
- Upon the successful completion of the Rabi experiment, run Pingpong amplitude calibration with default parameters.
'''

    description = description_rabi
    stages = get_stages_from_instruction(description)
    print(stages)