import warnings
from typing import Dict, Any, List

from k_agents.staging.stage_execution import check_if_needed_to_break_down, Stage, \
    get_exp_from_var_table
from k_agents.staging.stage_generation import get_stages_from_instruction, stages_to_html
from k_agents.notebook_utils import show_spinner, hide_spinner
from k_agents.staging import find_the_stage_label_based_on_description
from k_agents.staging.stage_transition import get_next_stage_label, \
    generate_new_stage_description
from k_agents.notebook_utils import display_chat, code_to_html, dict_to_html
from IPython.core.display import display, HTML

from k_agents.translation.agent import TranslationAgent, get_codegen_wm
from k_agents.translation.env import TranslationAgentEnv
from k_agents.experiment.experiment import Experiment
from k_agents.variable_table import VariableTable
from k_agents.indexer.code_indexer import build_leeq_code_ltm
import numpy

np = numpy
__all__ = ["OneInstExecutionAgent", "ExecutionAgent", "AutoRun"]
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
    input_var_table = VariableTable()
    for key, value in kwargs.items():
        input_var_table.add_variable(key, value)

    leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
    code_cog_model = TranslationAgent()
    for idea in leeq_code_ltm.ideas:
        code_cog_model.lt_memory.add_idea(idea)
    code_cog_model.n_recall_items = 5  # Number of items to recall in cognitive model
    var_table: VariableTable = VariableTable()
    var_table.add_parent_table(exps_var_table)
    var_table.add_parent_table(input_var_table)

    codegen_wm = get_codegen_wm(prompt, input_var_table)

    recall_res = code_cog_model.recall(codegen_wm)
    codes = code_cog_model.codegen(codegen_wm, recall_res)

    new_var_table = var_table.new_child_table()

    hide_spinner(spinner_id)
    code_html = code_to_html(codes)
    display_chat("Execution agent (generating code)", 'light_purple', f"Here is the generated code:<br>{code_html}")
    new_var_table.interpret(codes)
    return new_var_table




class StageExecutionAgent:
    """
    An experiment that contains multiple stages to be run.
    """

    def __init__(self):
        super().__init__()
        self.stages = None
        self.experiment_history = []
        self.final_result = None
        self.max_step_per_stage = 6

        trans_env = TranslationAgentEnv()
        self.translation_agent = trans_env.translation_agent
        self.translation_var_table = trans_env.translation_var_table
        assert self.translation_agent is not None, "Translation agent has not been initialized."


    def run(self, stages: List[Stage], sub_experiment=False, **kwargs):
        """
        Run the staged experiment powered by language model.

        Parameters
        ----------
        stages: List[Stage]
            The stages of the experiment.
        sub_experiment: bool
            Whether the experiment is a sub-experiment. If it is we do not allow it to be further splitted.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """

        self.stages: List[Stage] = stages
        exp_inputs_table, runtime_var_table = self.make_var_table(self.translation_var_table, kwargs)
        coding_ltm_cache = {}
        curr_stage = self.stages[0]

        self.experiment_history = []

        for step in range(len(self.stages) * self.max_step_per_stage):

            exp_object = run_stage_description(curr_stage, self.translation_agent, runtime_var_table, exp_inputs_table, coding_ltm_cache,
                                                  sub_experiment)

            if exp_object is None:
                warnings.warn(f"Experiment object not found in the variable table.")
                # re-run the stage
                continue

            self.experiment_history.append(exp_object)

            experiment_result = exp_object.get_ai_inspection_results()

            experiment_analysis_html = dict_to_html(experiment_result)
            color = 'light_green' if experiment_result['Experiment success'] else 'light_red'
            self.message_box(f"Experiment analysis results are as follows:<br>{experiment_analysis_html}", color=color)

            spinner_id = show_spinner(f"Considering the next stage...")

            next_stage_info = get_next_stage_label(curr_stage, experiment_result)
            next_stage_label = next_stage_info["next"]
            additional_info = next_stage_info["additional_info"]

            if next_stage_label in ["Complete", "Fail"]:
                hide_spinner(spinner_id)
                break

            next_stage = self.find_next_stage(next_stage_label)

            if curr_stage.label in next_stage.label:
                new_description = generate_new_stage_description(next_stage, additional_info)
                next_stage.description = new_description

            hide_spinner(spinner_id)

            self.message_box(f"Transitioning to the next stage {next_stage.label} "
                             f"with the following description:<br>"
                             f"{next_stage.description}<br>"
                             f"{next_stage_info['analysis']}")
            curr_stage = next_stage
        else:
            next_stage_label = "Too many steps"


        if next_stage_label == "Complete":
            self.message_box("The experiment is complete.<br>" + f"{next_stage_info['analysis']}", color='light_green')
        elif next_stage_label == "Fail":
            self.message_box("The experiment has failed.<br>" + f"{next_stage_info['analysis']}", color='light_red')
        elif next_stage_label == "Too many steps":
            self.message_box("Too many steps have been taken. The experiment is not complete.", color='light_red')

        self.final_result = {
            "success": next_stage_label == "Complete",
            "analysis": self.experiment_history[-1].get_ai_inspection_results()
        }

    def message_box(self, content, color='light_blue'):
        display_chat("Execution Agent", color, content)


    def find_next_stage(self, next_stage_label):
        next_stage: Stage
        for stage in self.stages:
            if stage.label in next_stage_label:
                next_stage = stage
                break
        else:
            next_stage = find_the_stage_label_based_on_description(self.stages,
                                                                   next_stage_label)
            if next_stage is None:
                assert False, f"Next stage label {next_stage_label} not found in stages"
        return next_stage

    def make_var_table(self, translation_var_table, kwargs):

        exp_inputs_table = VariableTable.from_dict(kwargs)

        var_table = VariableTable()
        var_table.add_parent_table(translation_var_table)
        var_table.add_parent_table(exp_inputs_table)

        return exp_inputs_table, var_table


    def get_experiment_history(self) -> List[Experiment]:
        """
        Get the history of experiments run in the staged experiment.

        Returns
        -------
        List[Experiment]: The history of experiments run in the staged experiment.
        """
        return self.experiment_history

    def get_last_experiment(self) -> Experiment:
        """
        Get the last experiment run in the staged experiment.

        Returns
        -------
        Experiment: The last experimen't run in the staged experiment.
        """
        return self.experiment_history[-1]

    def get_ai_inspection_results(self) -> Dict[str, Any]:
        return self.get_last_experiment().get_ai_inspection_results()

        # return {
        #    "Analysis": self.final_result["analysis"],
        #    "Suggested parameter updates": None,
        #    'Experiment success': self.final_result["success"],
        # }

def init_translation_agent():
    """
    Initialize the translation agent for the experiments
    """
    leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
    translation_agent = TranslationAgent()

    for idea in leeq_code_ltm.ideas:
        translation_agent.lt_memory.add_idea(idea)
    translation_agent.n_recall_items = 5

    moduler_var_table = VariableTable()
    moduler_var_table.add_variable("np", np)
    moduler_var_table.add_variable("numpy", np)

    translation_var_table = VariableTable()
    translation_var_table.add_parent_table(exps_var_table)
    translation_var_table.add_parent_table(moduler_var_table)

    env = TranslationAgentEnv()
    env.translation_agent = translation_agent
    env.translation_var_table = translation_var_table



def run_stage_description(stage: 'Stage', translation_agent, var_table, exp_inputs_table: VariableTable, coding_ltm_cache, sub_experiment):
    """
    Run the stage description powered by language model.

    Parameters
    ----------
    stage: Stage
        The stage to run.
    """
    spinner_id = show_spinner(f"Executing {stage.label}: {stage.title}...")

    prompt = f"""
    Overview of the funcationality: {stage.overview}
    Current stage: {stage.label}
    """

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
        exp = ExecutionAgent(stage.description, sub_experiment=True,
                             **exp_inputs_table.variable_objs)
        new_var_table = var_table.new_child_table()
        new_var_table.add_variable("exp", exp)

        return new_var_table

    codegen_wm = get_codegen_wm(stage.description, exp_inputs_table)

    if stage.title not in coding_ltm_cache:
        recall_res = translation_agent.recall(codegen_wm)
        coding_ltm_cache[stage.title] = recall_res
    else:
        recall_res = coding_ltm_cache[stage.title]

    # with display_chats():
    codes = translation_agent.codegen(codegen_wm, recall_res)

    new_var_table = var_table.new_child_table()

    hide_spinner(spinner_id)
    code_html = code_to_html(codes)
    display_chat("Execution agent (generating code)", 'light_purple',
                 f"Here is the generated code:<br>{code_html}")
    new_var_table.interpret(codes)

    exp_object = get_exp_from_var_table(new_var_table)

    return exp_object

class OneInstExecutionAgent(StageExecutionAgent):
    """
    An experiment that contains one instruction (step) to be run. The instructions are powered by language model.
    """

    def run(self, instructions: str, next_stage_guide=None, **kwargs):
        """
        Run the experiment powered by language model.

        Parameters
        ----------
        instructions: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """
        # label: str, title: str, overview: str, description: str, next_stage_guide: str

        if next_stage_guide is None:
            next_stage_guide = """Go to Complete if success. 
                                Go back to the same stage if the experiment failed and the parameters should be adjusted.
                                Go to Fail if the experiment failed and the parameters cannot be adjusted.
                                Go to Fail if the experiment failed and there is no suggestion for how to adjust the parameters.
                                Follow the instructions on how to transit to the next stage from the report of the experiment if there is any.
                                Go to Fail if the experiment has failed after 3 attempts."""

        stage = Stage(label="Stage1", title="Implement experiment",
                      overview='You are requested to implement one experiment and modify the parameter to make it success.',
                      description=instructions, next_stage_guide=next_stage_guide
                      )
        stage_complete = Stage("Complete", "Complete", "The experiment is complete.", "End of experiment.",
                               next_stage_guide='None')
        stage_fail = Stage("Fail", "Fail", "The experiment has failed.", "End of experiment.", next_stage_guide='None')
        stages = [stage, stage_complete, stage_fail]

        super().run(stages, **kwargs)


class ExecutionAgent(StageExecutionAgent):
    """
    A fully automated experiment that contains multiple steps. Automatically runs the experiment based on the instructions
    provided.
    """

    def run(self, instructions: str, sub_experiment=False, **kwargs):
        """
        Run the automated experiment powered by language model.

        Parameters
        ----------
        instructions: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        sub_experiment: bool
            Whether the experiment is a sub-experiment. If it is we do not allow it to be further splitted.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """


        spinner_id = show_spinner("AI is designing the experiment...")
        stages = get_stages_from_instruction(instructions)
        hide_spinner(spinner_id)
        self.message_box("The planned experiments are:<br>" + stages_to_html(stages), color='light_blue')

        super().run(stages, sub_experiment=sub_experiment, **kwargs)

def AutoRun(instructions, **kwargs):
    ExecutionAgent().run(instructions, **kwargs)

if __name__ == '__main__':
    from k_agents.ideanet.recall_logger import RecallLogger

    prompt = "Do qubit measurement calibration to update the GMM model."
    wm = get_codegen_wm(prompt, VariableTable())
    leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
    translation_agent = TranslationAgent()
    translation_agent.n_recall_items = 5
    for idea in leeq_code_ltm.ideas:
        translation_agent.lt_memory.add_idea(idea)

    with RecallLogger():
        code = translation_agent.codegen(wm)