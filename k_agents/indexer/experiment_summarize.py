from mllm import Chat

from k_agents.experiment.experiment import Experiment, _rebuild_args_dict
from k_agents.notebook_utils import show_spinner, hide_spinner


def get_experiment_summary(exp: Experiment, results_list: list):
    """
    Summarize the experiment results.
    """
    arg_dict = _rebuild_args_dict(exp.run, exp.run_args, exp.run_kwargs)
    results_str = "".join([f"{i}: {result}" + '\n' for i, result in enumerate(results_list)])
    prompt = f"""
Summarize the experiment results and report the key results. Indicate if the experiment was successful or failed.
If failed, suggest possible updates to the parameters or the experiment design if the experiment fails. The suggestion
needs to be specific on how much of the quantity needs to be changed on the parameters. Otherwise return None for the
parameter updates.

<Experiment document>
{exp.run.__doc__}
</Experiment document>

<Run parameters>
{arg_dict}
</Run parameters>

<Experiment description> 
{exp._experiment_result_analysis_instructions}
</Experiment description>

<Results>
{results_str}
</Results>

Return in json format:
<Return>
{{
    "analysis": str,
    "parameter_updates": "<parameter 1> should be updated by <value> and <parameter 2> should be updated by <value>",
    "success": bool,
}}
</Return>
"""

    spinner_id = show_spinner(f"Analyzing experiment results...")
    chat = Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", expensive=True, cache=True)
    hide_spinner(spinner_id)
    return res
