import json
import pprint
import warnings
from typing import Dict, Any, List, TYPE_CHECKING

import numpy
from IPython.core.display import display, HTML
from mllm import Chat

from k_agents.experiment.experiment import Experiment
from k_agents.notebook_utils import display_chat, code_to_html, dict_to_html
from k_agents.notebook_utils import show_spinner, hide_spinner
from k_agents.execution import find_the_stage_label_based_on_description
from k_agents.execution.stage_execution import check_if_needed_to_break_down, Stage, \
    get_exp_from_var_table
from k_agents.execution.stage_generation import get_stages_from_instruction, \
    stages_to_html
from k_agents.execution.stage_transition import get_next_stage_label, \
    generate_new_stage_description
from k_agents.translation.agent import get_codegen_wm
from k_agents.translation.env import TranslationAgentEnv
from k_agents.variable_table import VariableTable

np = numpy
__all__ = ["AutomatedExperiment", "AutoRun"]


def add_inspection_result_to_var_table(inspection_result, runtime_var_table):
    if not isinstance(inspection_result, dict):
        return
    for key, value in inspection_result.items():
        if key.lower() in ["success", "analysis"]:
            continue
        elif isinstance(value, str):
            continue
        elif value is None:
            continue
        else:
            key = key.replace(" ", "_")
            key = key.replace("-", "_")
            runtime_var_table.add_variable("_"+key, value)


class ExecutionAgent:
    """
    An experiment that contains multiple stages to be run.
    """

    def __init__(self):
        super().__init__()
        self.stages: List[Stage] = None
        self.history_experiments: List[Experiment] = []
        self.history_inspections: List[Dict[str, Any]] = []
        self.succeeded = None
        self.final_analysis = None
        self.max_step_per_stage = 6

        trans_env = TranslationAgentEnv()
        self.translation_agent = trans_env.translation_agent
        self.translation_var_table = trans_env.translation_var_table
        assert self.translation_agent is not None, "Translation agent has not been initialized."

    def generate_stages(self, steps: str):
        """
        Generate stages from the instructions.

        Parameters
        ----------
        steps: str
            The instructions to generate the stages.

        Returns
        -------
        stages: List[Stage]
            The generated stages.
        """
        spinner_id = show_spinner("Generating state machine...")
        self.stages = get_stages_from_instruction(steps)
        hide_spinner(spinner_id)
        agent_message_box(
            "The planned experiments are:<br>" + stages_to_html(self.stages),
            color='light_blue')

    def run(self, variables, sub_experiment=False):
        """
        Run the staged experiment powered by language model.

        Parameters
        ----------
        sub_experiment: bool
            Whether the experiment is a sub-experiment. If it is we do not allow it to be further splitted.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """
        assert self.stages is not None, "Stages have not been generated. Rub self.generate_stages first."
        exp_inputs_table, runtime_var_table = make_var_table(self.translation_var_table,
                                                             variables)
        coding_ltm_cache = {}
        curr_stage = self.stages[0]

        self.history_experiments = []

        for step in range(len(self.stages) * self.max_step_per_stage):
            curr_stage.n_failed += 1
            exp_object = run_stage_description(curr_stage, self.translation_agent,
                                               runtime_var_table, exp_inputs_table,
                                               coding_ltm_cache,
                                               True)

            if exp_object is None:
                warnings.warn(f"Experiment object not found in the variable table.")
                # re-run the stage
                continue

            inspection_result = exp_object.get_ai_inspection_summary()

            add_inspection_result_to_var_table(inspection_result, runtime_var_table)

            self.history_experiments.append(exp_object)
            self.history_inspections.append(inspection_result)

            experiment_analysis_html = dict_to_html(inspection_result)
            color = 'light_green' if inspection_result[
                'success'] else 'light_red'
            agent_message_box(
                f"Experiment analysis results are as follows:<br>{experiment_analysis_html}",
                color=color)

            spinner_id = show_spinner(f"Considering the next stage...")

            next_stage_info = get_next_stage_label(curr_stage, inspection_result)
            next_stage_label = next_stage_info["next"]

            if next_stage_label in ["Complete", "Fail"]:
                hide_spinner(spinner_id)
                break

            next_stage = find_next_stage(self.stages, next_stage_label)

            if curr_stage.label in next_stage.label:
                #new_description = generate_new_stage_description(next_stage)
                #next_stage.description = new_description
                ...

            hide_spinner(spinner_id)

            agent_message_box(f"Transitioning to the next stage {next_stage.label} "
                              f"with the following description:<br>"
                              f"{next_stage.description}<br>"
                              f"{next_stage_info['analysis']}")
            curr_stage = next_stage
        else:
            next_stage_label = "Too many steps"

        if next_stage_label == "Complete":
            agent_message_box(
                "The experiment is complete.<br>" + f"{next_stage_info['analysis']}",
                color='light_green')
        elif next_stage_label == "Fail":
            agent_message_box(
                "The experiment has failed.<br>" + f"{next_stage_info['analysis']}",
                color='light_red')
        elif next_stage_label == "Too many steps":
            agent_message_box(
                "Too many steps have been taken. The experiment is not complete.",
                color='light_red')

        self.succeeded = next_stage_label == "Complete"
        self.final_analysis = self.history_experiments[-1].get_ai_inspection_summary()

    def history_to_prompt(self):
        prompt = []
        for inspection_result in self.history_inspections:
            prompt.append(json.dumps(inspection_result, indent=0))
        return "\n".join(prompt)


def find_next_stage(stages, next_stage_label):
    next_stage: Stage
    for stage in stages:
        if stage.label in next_stage_label:
            next_stage = stage
            break
    else:
        next_stage = find_the_stage_label_based_on_description(stages,
                                                               next_stage_label)
        if next_stage is None:
            assert False, f"Next stage label {next_stage_label} not found in stages"
    return next_stage


def agent_message_box(content, color='light_blue'):
    display_chat("Execution Agent", color, content)


def make_var_table(translation_var_table, kwargs):
    exp_inputs_table = VariableTable.from_dict(kwargs)

    var_table = VariableTable()
    var_table.add_parent_table(translation_var_table)
    var_table.add_parent_table(exp_inputs_table)

    return exp_inputs_table, var_table


def run_stage_description(stage: 'Stage', translation_agent, runtime_var_table,
                          exp_inputs_table: VariableTable, coding_ltm_cache,
                          sub_experiment):
    """
    Run the stage description powered by language model.

    Parameters
    ----------
    stage: Stage
        The stage to run.
    """
    spinner_id = show_spinner(f"Executing {stage.label}: {stage.title}...")

    html = stages_to_html([stage])
    display(HTML(html))

    if sub_experiment:
        single_step = True
    else:
        breakdown_requirement = check_if_needed_to_break_down(stage.description)
        single_step = breakdown_requirement['single_step'] or len(
            breakdown_requirement['steps']) == 1

    if not single_step:
        hide_spinner(spinner_id)
        display_chat("Stage Planning AI", 'light_blue',
                     f"Stage {stage.label} is too complex to be processed in one step. Planning to break down the stage into smaller steps. {breakdown_requirement['reason']}.")
        exp = AutomatedExperiment(stage.description, runtime_var_table.variable_objs)
        new_var_table = runtime_var_table.new_child_table()
        new_var_table.add_variable("exp", exp)

        return new_var_table

    prompt_table = VariableTable()
    prompt_table.update_by_other_table(runtime_var_table)
    prompt_table.update_by_other_table(exp_inputs_table)
    codegen_wm = get_codegen_wm(stage.description, prompt_table)

    if stage.title not in coding_ltm_cache:
        recall_res = translation_agent.recall(codegen_wm)
        coding_ltm_cache[stage.title] = recall_res
    else:
        recall_res = coding_ltm_cache[stage.title]

    # with display_chats():
    codes = translation_agent.codegen(codegen_wm, recall_res)

    new_var_table = runtime_var_table.new_child_table()

    hide_spinner(spinner_id)
    code_html = code_to_html(codes)
    display_chat("Execution agent (generating code)", 'light_purple',
                 f"Here is the generated code:<br>{code_html}")
    new_var_table.interpret(codes)

    exp_object = get_exp_from_var_table(new_var_table)

    return exp_object


class AutomatedExperiment(Experiment):
    """
    A fully automated experiment that contains multiple steps. Automatically runs the experiment based on the instructions
    provided.
    """

    _title = None
    _original_steps = None
    _background = None
    _expected_result = None
    decorate_run = False

    def __init__(self, instructions: str, variables):
        super().__init__()
        self.exec_agent: ExecutionAgent = ExecutionAgent()
        self.instructions = instructions
        self.run(instructions, variables)

    def run(self, instructions: str, variables: dict, sub_experiment=False, ):
        """
        Run the automated experiment powered by language model.

        Parameters
        ----------
        instructions: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        sub_experiment: bool
            Whether the experiment is a sub-experiment. If it is we do not allow it to be further splitted.
        variables: dict

        Returns
        -------
        """
        if self._original_steps is not None:
            spinner_id = show_spinner(f"Decomposing instructions...")
            steps = self.decompose_instruction(instructions)
            hide_spinner(spinner_id)
        else:
            steps = instructions

        self.exec_agent.generate_stages(steps)
        self.exec_agent.run(variables, sub_experiment=sub_experiment)

    def decompose_instruction(self, instruction):
        prompt = f"""
        You are required to modify a instruction based on an example.
        <example>
        <instruction>
        {self._title}
        </instruction>
        <steps>
        {self._original_steps}
        </steps>
        <example>
        The instruction has the following background description, which might be irrelevant your task:
        <background>
        {self._background}
        </background>
        Your task is to modify the following input_instruction into its corresponding steps in a way that is similar to the example.
        <input_instruction>
        {instruction}
        </input_instruction>
        <requirements>
        You are required to output a JSON dict with keys:
        - "analysis" (string): An analysis of how to decompose the input_instruction into steps.
        - "steps" (string): A string for the decomposed steps of the input_instruction based on <steps>. You should make minimal modifications to the <steps> when adopt it to the new steps.
        """
        chat = Chat(prompt, dedent=True)
        res = chat.complete(parse="dict", expensive=True)
        return res["steps"]

    def get_ai_inspection_summary(self):
        assert self.exec_agent.succeeded is not None, "Experiment has not been run yet."
        history = self.exec_agent.history_to_prompt()
        expected_result_instruction_prompt = ""
        if self._expected_result is not None:
            expected_result_instruction_prompt = f"""
            You need to emphasize the expected key results of the experiment.
            <report_instruction>
            {self._expected_result}
            </report_instruction>
            """

        success_prompt = "The experiment has succeeded." if self.exec_agent.succeeded else "The experiment has failed."

        prompt = f"""
        You are trying to summarize the execution of an experiment that contains multiple stages. The experiment has been run and the results have been recorded. 
        The experiment is described as follows:
        <title>
        {self._title}
        </title>
        <background>
        {self._background}
        </background>
        <steps>
        {self.instructions}
        </steps>
        
        {success_prompt} The report of each steps of the experiment are as follows:
        <results>
        {history}
        </results>
         
        {expected_result_instruction_prompt}
        
        <requirements>
        You are required to output a JSON dict with keys:
        - "analysis" (string): An analysis of the results of the experiment.
        - "results" (string): A very short summary of the experiment with an emphasis on the key results.
        </requirements>
        """
        chat = Chat(prompt, dedent=True)
        res = chat.complete(parse="dict")
        return {
            "results": res["results"],
            "success": self.exec_agent.succeeded,
        }


def AutoRun(instruction, **kwargs):
    exp = AutomatedExperiment(instruction, kwargs)
    return exp


def execute_experiment_from_prompt(prompt: str, **kwargs):
    """
    Execute an experiment from a prompt.

    Parameters
    ----------
    prompt: str
        The prompt to run the experiment.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    The variable table after the experiment is run.

    """

    spinner_id = show_spinner(f"Interpreting experiment...")

    translation_agent_env = TranslationAgentEnv()
    translation_agent = translation_agent_env.translation_agent
    translation_var_table = translation_agent_env.translation_var_table

    input_var_table = VariableTable.from_dict(kwargs)
    var_table: VariableTable = VariableTable()
    var_table.add_parent_table(translation_var_table)
    var_table.add_parent_table(input_var_table)

    codegen_wm = get_codegen_wm(prompt, input_var_table)

    recall_res = translation_agent.recall(codegen_wm)
    codes = translation_agent.codegen(codegen_wm, recall_res)

    new_var_table = var_table.new_child_table()

    hide_spinner(spinner_id)
    code_html = code_to_html(codes)
    display_chat("Execution agent (generating code)", 'light_purple',
                 f"Here is the generated code:<br>{code_html}")
    new_var_table.interpret(codes)
    return new_var_table
