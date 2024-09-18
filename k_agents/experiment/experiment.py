import functools
import inspect
from typing import Union, Callable, Any, Dict, List

from k_agents.indexer.experiment_summarize import get_experiment_summary
from k_agents.notebook_utils import show_spinner, hide_spinner, dict_to_html, display_chat
from k_agents.vlms import has_visual_analyze_prompt, get_visual_analyze_prompt, \
    visual_inspection


class Experiment:
    _experiment_result_analysis_instructions = None

    def __init__(self, *args, **kwargs):
        self._ai_inspection_results = {}
        self._ai_final_analysis = None
        self._plot_function_result_objs = {}
        self._plot_function_images = {}
        # A log of the arguments and keyword arguments of the last run.
        self.run_args = None
        self.run_kwargs = None

        # warp the run method as _run using functools.wraps
        self.bare_run = self.run
        self.bare_run_simulated = self.run_simulated
        self._decorate_run()


    def _decorate_run(self):
        self.run = self._run
        self.run.__dict__["__qualname__"] = self.bare_run.__qualname__
        self.run.__dict__["__doc__"] = self.bare_run.__doc__
        self.run_simulated = self._run_simulated
        self.run_simulated.__dict__["__qualname__"] = self.bare_run_simulated.__qualname__
        self.run_simulated.__dict__["__doc__"] = self.bare_run_simulated.__doc__

    def get_run_args_dict(self):
        assert self.run_args is not None, "The experiment has not been run yet."
        assert self.run_kwargs is not None, "The experiment has not been run yet."
        return _rebuild_args_dict(self.bare_run, self.run_args, self.run_kwargs)


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


    def run_ai_inspection(self):
        for name, func in self.get_visual_inspection_functions():
            inspect_answer = self._run_ai_inspection_on_single_function(func)
            if inspect_answer is not None:
                color = 'light_green' if inspect_answer['success'] else 'light_red'
                html = dict_to_html(inspect_answer)

                display_chat(agent_name=f"Inspection AI",
                             content='<br>' + html,
                             background_color=color)

    def _run_ai_inspection_on_single_function(self, func):
        """
        Run the AI inspection on a single function.

        Parameters:
            func (callable): The function to analyze.
        Returns:
            dict: The result of the analysis.
        """

        if self._ai_inspection_results.get(func.__qualname__) is not None:
            return self._ai_inspection_results.get(func.__qualname__)

        try:
            if has_visual_analyze_prompt(func):
                if self._plot_function_images.get(func.__qualname__) is None:
                    self._execute_single_plot_function(func, build_static_image=True)

                image = self._plot_function_images.get(func.__qualname__)
                prompt = get_visual_analyze_prompt(func)

                spinner_id = show_spinner(f"Vision AI is inspecting the plots...")

                inspect_answer = visual_inspection(image, prompt, func._file_path)

                hide_spinner(spinner_id)

                self._ai_inspection_results[func.__qualname__] = inspect_answer

                return inspect_answer

            else:
                # Currently, only visual inspection is supported.
                # However, it is possible to add more inspection methods here.
                raise ValueError(
                    f"Function {func.__qualname__} is not an AI inspection function."
                )

        except Exception as e:
            self.log_warning(
                f"Error when running single AI inspection on {func.__qualname__} (Ignore the error and continue): {e} ")

        return None

    def _execute_plot_functions(self, build_static_image=False):
        """
        Execute the browsable plot function.

        Parameters:
            build_static_image (bool): Whether to build the static image.
        """
        for name, func in self.get_plot_functions():
            try:
                self._execute_single_plot_function(func,
                                                   build_static_image=build_static_image)
            except Exception as e:
                msg = f"Error when executing the browsable plot function {name}:{e}."
                self.log_warning(msg)

    @classmethod
    def get_plot_functions(cls):
        tag_names = ["_browser_function", "_is_plot_function"]
        tagged_methods = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if any([getattr(method, tag_name, None) for tag_name in tag_names]):
                tagged_methods.append((name, method))
        return tagged_methods

    def get_analyzed_result_prompt(self) -> Union[str, None]:
        """
        Get the natual language description of the analyzed result for AI.

        Returns
        -------
        str: The prompt to analyze the result.
        """
        return None

    @staticmethod
    def _build_static_image(result):
        from leeq.utils.ai.utils import matplotlib_plotly_to_pil
        return matplotlib_plotly_to_pil(result)

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
            image = self._build_static_image(figure_obj)
            self._plot_function_images[func.__qualname__] = image

        self._plot_function_result_objs[func.__qualname__] = figure_obj


    def get_ai_inspection_results(self, inspection_method='full', ignore_cache=False):
        """
        Get the AI inspection results.

        Parameters:
            inspection_method (str): The inspection method to use. Can be 'full' or 'visual_only' or 'fitting_only'.
            ignore_cache (bool): Whether to ignore the cache.

        Returns:
            dict: The AI inspection results.
        """
        ai_inspection_results = {}

        assert inspection_method in ['full', 'visual_only', 'fitting_only'], \
            f"inspection_method must be 'full', 'visual_only' or 'fitting_only', got {inspection_method}"

        # Add the visual inspection results to the AI inspection results.
        if inspection_method != 'fitting_only':
            # Ensure that all visual inspection functions are executed.
            for name, func in self.get_visual_inspection_functions():
                if self._ai_inspection_results.get(func.__qualname__) is None:
                    self._run_ai_inspection_on_single_function(func)
            # Add the visual inspection results to the AI inspection results.
            ai_inspection_results = {key.split('.')[-1]: val['analysis'] for key, val
                                     in
                                     self._ai_inspection_results.items()}

        # Add the fitting results to the AI inspection results.
        if inspection_method != 'visual_only':
            fitting_results = self.get_analyzed_result_prompt()
            if fitting_results is not None:
                ai_inspection_results['fitting'] = fitting_results

        # Summarize the AI inspection results based on the experiment result analysis instructions.
        if self._experiment_result_analysis_instructions is None:
            raise ValueError(
                "The experiment result analysis instructions are not defined."
            )

        if self._ai_final_analysis is None or ignore_cache:
            spinner_id = show_spinner(
                f"AI is analyzing the experiment results...")

            arg_dict = _rebuild_args_dict(self.run, self.run_args, self.run_kwargs)
            run_args_prompt = \
f"""
Document of this experiment:
{self.run.__doc__}
Running arguments:
{arg_dict}
"""

            summary = get_experiment_summary(
                self._experiment_result_analysis_instructions, run_args_prompt,
                ai_inspection_results)

            if not ignore_cache:
                self._ai_final_analysis = summary

            hide_spinner(spinner_id)
        else:
            summary = self._ai_final_analysis

        ai_inspection_results['Final analysis'] = summary['analysis']
        ai_inspection_results['Suggested parameter updates'] = summary[
            'parameter_updates']
        ai_inspection_results['Experiment success'] = summary['success']

        return ai_inspection_results



    @classmethod
    def get_visual_inspection_functions(cls):
        tagged_methods = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, "_visual_prompt", None):
                tagged_methods.append((name, method))
        return tagged_methods

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
        if self.to_run_ai_inspection:
            self.run_ai_inspection()

    def log_warning(self, message):
        raise NotImplementedError()

    def log_info(self):
        raise NotImplementedError()

    @classmethod
    def is_ai_compatible(cls):
        """
        A method to indicate that the experiment is AI compatible.
        """
        return cls._experiment_result_analysis_instructions is not None

    @property
    def is_simulation(self):
        return False

    @property
    def to_run_ai_inspection(self):
        return True

    @property
    def to_show_figure_in_notebook(self):
        return True


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

