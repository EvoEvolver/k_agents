from typing import List

from mllm.utils import parallel_map

from k_agents.execution.stage_execution import Stage
import json
import mllm


def stages_to_html(stages_list: List[Stage]):
    stages_dict = {stage.label: stage.to_dict() for stage in stages_list}

    html_content = '<div style="font-family: Arial, sans-serif;">'

    # Loop through each stage in the dictionary
    for stage_key, stage_info in stages_dict.items():
        # Skip "Complete" and "Fail" stages
        if stage_key in ["Complete", "Fail"]:
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
You are required to separate a description based on some rules.

<reference>
{res['Reference']}
</reference>

<title>
{res['Title']}
</title>

<description>
{res['ExperimentDescription']}
</description>

<requirements>
- If the description contains information not related to the input, remove it. 
- If the description contains objectives or goals, remove them.
- Quote the the parameters and the values in the format of `"<parameter name>=<parameter value>"` if the actual values are present in the description. 
    The values should be the actual values, not placeholders. 
- Only modify the parts described above, keep the rest of the description as is.
- If this stage only contains data and result analysis and interpretation without carrying out any experiment,
  please set the <contains_experiment> to False. Otherwise set it to True.
- If the description contains information that is not present in the reference, for example the details
   how to implement each stages but they are not specifically described in the reference, remove these information.
- Do not include any additional information in the description. Do not include any information that is not presented in the description, such as the details
 in how to implement each steps based on your knowledge.
</requirements>

<formats>
Response in JSON with the following keys:
"analysis" (string):" an analysis about how to update the stage description.,
"action_description" (string): the description about the action. For example: "Conduct the <experiment name> with xxx parameter." You should not mention what to do next based on the result.
"background_description" (string): the background information of the experiment. (could be empty),
"next" (string): what to do next based on the result of the experiment. (could be empty),
"contains_experiment" (bool): whether the stage contains an experiment or not.
</formats>
"""

    chat = mllm.Chat(prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict")
    updated_res = chat.complete(parse="dict", expensive=True, cache=True)

    new_res = {
        "Title": res["Title"],
        "ExperimentDescription": updated_res["action_description"],
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
<objective>
Your objective is to divide the experimental description into stages. Each stage of the experiment should represent a distinct operation with explicit instructions and transition rules. The stages must be concise, self-contained, and clearly defined. Especially, you are required to output a JSON dict with the following elements:
</requirements>
<output_format>

You should output a JSON dict containing keys for each stage of the experiment. The key should be the label of the stage. The value should be a dict containing the following keys:

- Title: a descriptive title for the stage.

- ExperimentDescription: a procedural outline for each stage of the experiment. The description should explicitly state the name of the experiment, list all parameters involved, and clearly outline the step to be taken. You should not mention how the experiment will be executed.

- Next: a description of the transition rules to proceed to the next stage based on the results of the experiment. This should be a clear and concise instruction on how to advance to the next stage.
Note: By default, always proceed to the next stage when the experiment succeeded.
Note: When there are additional descriptions about how to transition to the next stage based on the results of the experiment, include them in the transition rules.

- Reference: The original part of <experiment_description> that is related to your experiment description.
</output_format>
""" + """
<output_example>
{
  "Stage1": {
    "Title": "Experiment1",
    "ExperimentDescription": "Conduct the <experiment name 1> with parameters <parameter list for experiment 1>.",
    "Next": "Proceed to Stage ... if successful. Else, proceed to Stage ..."
    "Reference":'<The original input prompt related to this stage>'
  },
  "Stage ...": {
    "Title": "Experiment ...",
    "ExperimentDescription": "Conduct the <experiment name ...> with parameters <parameter list for experiment ...>.",
    "Next": "Proceed to Complete if xx. Proceed to Stage ... if xxx"
    "Reference":'<The original input prompt related to this stage>'
  },
  "Complete": {},
  "Fail": {}
}
</output_example>
<Notice>
- You should divide the description into distinct stages, each representing a specific operation.
- Do not include any additional information that is not present in the description. You must not imagine the details how to implement each stages.
- The description might mixes the action description and transition rules, you must separate them. You must not take a transition rule as a separate stage. 
- The Next key must be a string detailing the transition conditions. Do not use "retry", or "revert", instead describe the stage label directly.
- Generate as less stages as possible, ideally just one stage. However, you must make sure each stage is distinct and does not contain more than one experiment to carry out.
- If the description is very short, you must not adding extra information beyond the original description.
</Notice>
"""

    completed_prompt = prompt

    chat = mllm.Chat(completed_prompt,
                     "You are a very smart and helpful assistant who only reply in JSON dict", dedent=True)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    stages = []

    meta_stages = {
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
    res.update(meta_stages)

    # Add overview to each dict in res
    for stage_name, stage_content in res.items():
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

    stages_info_lines = []
    for stage in stages:
        stages_info_lines.append(f"- {stage.label}")
    stages_info_lines.append("- Complete")
    stages_info_lines.append("- Fail")
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
    for stage in stages:
        print(stage.to_dict())
