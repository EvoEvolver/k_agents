import inspect
import warnings
from typing import Callable, Any, Dict, List

from mllm import Chat

from k_agents.inspection.agents import InspectionAgent
from k_agents.inspection.vlms import matplotlib_plotly_to_pil
from k_agents.notebook_utils import show_spinner, hide_spinner


class Experiment:
    _experiment_result_analysis_instructions = None
    _rewrite_json_requirement = False
    decorate_run = True

    def __init__(self, *args, **kwargs):
        self._ai_inspection_results = {}
        self._ai_inspection_summary = None

        self._plot_function_result_objs = {}
        self._plot_function_images = {}

        # A log of the arguments and keyword arguments of the last run.
        self.run_args = None
        self.run_kwargs = None

        # warp the run method as _run using functools.wraps
        self.bare_run = self.run
        self.bare_run_simulated = self.run_simulated
        if self.decorate_run:
            self._decorate_run()

    def _decorate_run(self):
        self.run = self._run
        self.run_simulated = self._run_simulated

    def _run(self, *args, **kwargs):
        self._before_run(args, kwargs)
        try:
            if self.is_simulation:
                self.bare_run_simulated(*args, **kwargs)
            else:
                self.bare_run(*args, **kwargs)
        finally:
            self._post_run(args, kwargs)

    def _run_simulated(self, *args, **kwargs):
        self._before_run(args, kwargs)
        try:
            self.bare_run_simulated(*args, **kwargs)
        finally:
            self._post_run(args, kwargs)

    def run_simulated(self, *args, **kwargs):
        """
        Run the experiment in simulation mode. This is useful for debugging.
        """
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        """
        The main experiment script. Should be decorated by `labchronicle.log_and_record` to log the experiment.
        """
        raise NotImplementedError()

    def _before_run(self, args, kwargs):
        """
        Pre run method to be called before the experiment is run.
        """
        assert self.run_args is None, "Each instance of Experiment should only run once."
        assert self.run_kwargs is None, "Each instance of Experiment should only run once."
        self.run_args = args
        self.run_kwargs = kwargs

    def _post_run(self, args, kwargs):
        """
        Post run method to be called after the experiment is run.
        """
        # if self.to_run_ai_inspection:
        #    self.run_ai_inspection()
        ...

    def _get_run_args_dict(self):
        assert self.run_args is not None, "The experiment has not been run yet."
        assert self.run_kwargs is not None, "The experiment has not been run yet."
        return _rebuild_args_dict(self.bare_run, self.run_args, self.run_kwargs)

    def get_ai_inspection_summary(self):
        """
        Get the AI inspection results.

        Returns:
            dict: The AI inspection results.
        """
        agents = self._get_inspection_agents()
        summary = self._get_ai_inspection_results_from_some_agents(agents)
        return summary

    def _get_ai_inspection_results_from_some_agents(self, agents: List[InspectionAgent]):
        ai_inspection_results = {}
        # Add the visual inspection results to the AI inspection results.
        for agent in agents:
            inspect_answer = agent.run(self)
            if inspect_answer is not None:
                ai_inspection_results[agent.label] = inspect_answer

        self._ai_inspection_results = ai_inspection_results

        summary = self._summarize_inspection_results(ai_inspection_results)

        return summary

    def _summarize_inspection_results(self, ai_inspection_results: Dict[str, Dict]):
        # Summarize the AI inspection results based on the experiment result analysis instructions.

        assert self._experiment_result_analysis_instructions is not None, "The experiment result analysis instructions are not defined."
        arg_dict = _rebuild_args_dict(self.run, self.run_args, self.run_kwargs)

        if len(self._experiment_result_analysis_instructions) == 0:
            analysis_instruction_prompt = ""
        else:
            analysis_instruction_prompt = f"<analysis_instructions>\n{self._experiment_result_analysis_instructions}\n</analysis_instructions>"

        insp_result_in_prompt = []
        for label, result_dict in ai_inspection_results.items():
            insp_result_in_prompt.append(f"# {label}")
            for key, value in result_dict.items():
                insp_result_in_prompt.append(f"{key}: {value}")
            insp_result_in_prompt.append("")
        insp_results_str = "\n".join(insp_result_in_prompt)

        prompt = f"""
        Summarize the experiment results and report the key results. Indicate if the experiment was successful or failed.
    
        <Run parameters>
        {arg_dict}
        </Run parameters>
    
        <Results>
        {insp_results_str}
        </Results>
    
        {analysis_instruction_prompt}
        """
        if not self._rewrite_json_requirement:
            prompt += """
        <Return>
        Return in JSON with the following keys:
        "analysis" (string): an analysis of the experiment results and the success of the experiment.
        "success" (bool): whether the experiment was successful or not.
        </Return>        
        """

        spinner_id = show_spinner(f"Analyzing experiment results...")
        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict",
                    dedent=True)
        res = chat.complete(parse="dict", expensive=True, cache=True)
        hide_spinner(spinner_id)

        self._ai_inspection_summary = res

        return res

    @classmethod
    def _get_inspection_agents(cls):
        agents = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            agent = getattr(method, "_inspection_agent", None)
            if agent is not None:
                agents.append(agent)
        return agents

    def log_warning(self, message):
        warnings.warn(message)

    def log_info(self, message):
        print(message)

    @property
    def is_simulation(self):
        return False

    @property
    def to_run_ai_inspection(self):
        return True

    @property
    def to_show_figure_in_notebook(self):
        return True

    def _execute_plot_functions(self, build_static_image=False):
        """
        Execute the browsable plot function.

        Parameters:
            build_static_image (bool): Whether to build the static image.
        """
        for name, func in self._get_plot_functions():
            try:
                self._execute_single_plot_function(func,
                                                   build_static_image=build_static_image)
            except Exception as e:
                msg = f"Error when executing the browsable plot function {name}:{e}."
                self.log_warning(msg)

    @classmethod
    def _get_plot_functions(cls):
        tag_names = ["_browser_function", "_is_plot_function"]
        tagged_methods = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if any([getattr(method, tag_name, None) for tag_name in tag_names]):
                tagged_methods.append((name, method))
        return tagged_methods

    def _execute_single_plot_function(self, func: callable,
                                      build_static_image=False):
        """
        Execute the browsable plot function. The result and image will be stored in the function object
        attributes '_result' and '_image'.

        Parameters:
            func (callable): The browsable plot function.
            build_static_image (bool): Whether to build the static image.

        """
        figure_obj = self._plot_function_result_objs.get(func.__qualname__, None)

        if figure_obj is None:
            figure_obj = func(self)

        if build_static_image:
            image = matplotlib_plotly_to_pil(figure_obj)
            self._plot_function_images[func.__qualname__] = image

        self._plot_function_result_objs[func.__qualname__] = figure_obj


def _rebuild_args_dict(
        func: Callable[..., Any], called_args: List[Any], called_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Reconstruct the arguments dictionary for a given function based on its signature and the called arguments.

    This method fetches the signature of the function and tries to match the provided called arguments and
    keyword arguments to build the true arguments dictionary for the function call.

    Note that we have removed "self" from the called args, so be careful to remove it when it's a class method.

    Parameters:
    - func (Callable[..., Any]): The function for which the arguments dictionary needs to be built.
    - called_args (List[Any]): List of arguments with which the function was called.
    - called_kwargs (Dict[str, Any]): Dictionary of keyword arguments with which the function was called.

    Returns:
    - Dict[str, Any]: Dictionary containing the true arguments for the function call.

    Raises:
    - Exception: If there is a mismatch between the function's default arguments and its signature.
    """
    sig = inspect.signature(func)
    parameters = list(sig.parameters.values())

    mapped_args = {}

    # Remove "self" from the parameters if it's a class method
    if parameters[0].name == "self" or parameters[0].name == "cls":
        parameters = parameters[1:]

    # First, populate with defaults and arguments provided
    for param in parameters:
        if param.default != param.empty:  # if a default is provided
            mapped_args[param.name] = param.default
        else:
            mapped_args[param.name] = None

    # For positional arguments
    for param, value in zip(parameters, called_args):
        mapped_args[param.name] = value

    # For keyword arguments (this will override any values set before)
    mapped_args.update(called_kwargs)

    return mapped_args
