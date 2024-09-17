import inspect
from typing import Union

import matplotlib
import plotly
from IPython.display import display

from k_agents.indexer.experiment_summarize import get_experiment_summary
from k_agents.notebook_utils import show_spinner, hide_spinner
from k_agents.vlms import has_visual_analyze_prompt, get_visual_analyze_prompt, \
    visual_inspection
from leeq.utils.ai.display_chat.notebooks import dict_to_html, display_chat


class Experiment:
    _experiment_result_analysis_instructions = None

    def __init__(self, *args, **kwargs):
        self._ai_inspection_results = {}
        self._ai_final_analysis = None
        self._browser_function_results = {}
        self._browser_function_images = {}

    def _check_arguments(self, func, *args, **kwargs):
        """
        Check the arguments of the function.

        Parameters:
            func (callable): The function to check.
            args (list): The arguments of the function.
            kwargs (dict): The keyword arguments of the function.

        Returns:
            dict: The arguments of the function.
        """
        sig = inspect.signature(func)

        if 'ai_inspection' in kwargs and 'ai_inspection' not in sig.parameters:
            del kwargs['ai_inspection']

        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            msg = None
        except TypeError as e:
            msg = f"{e}\n\n"
            msg += f"Function signature: {sig}\n\n"
            msg += f"Documents:\n\n {self.run.__doc__}\n\n"

        if msg is not None:
            raise TypeError(msg)

        return bound


    def _run(self, bound):
        self._before_run()

        try:
            if self.is_simulation:
                self.run_simulated(*bound.args, **bound.kwargs)
            else:
                self.run(*bound.args, **bound.kwargs)
        finally:
            self._post_run()


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
                if self._browser_function_images.get(func.__qualname__) is None:
                    self._execute_single_browsable_plot_function(func,
                                                                 build_static_image=True)

                image = self._browser_function_images.get(func.__qualname__)

                spinner_id = show_spinner(f"Vision AI is inspecting the plots...")

                prompt = get_visual_analyze_prompt(func)
                inspect_answer = visual_inspection(image, prompt, func._file_path)
                self._ai_inspection_results[func.__qualname__] = inspect_answer

                hide_spinner(spinner_id)
                return inspect_answer

        except Exception as e:
            self.log_warning(
                f"Error when running single AI inspection on {func.__qualname__} (Ignore the error and continue): {e} ")

        return None

    def _execute_browsable_plot_function(self, build_static_image=False):
        """
        Execute the browsable plot function.

        Parameters:
            build_static_image (bool): Whether to build the static image.
        """
        for name, func in self.get_browser_functions():
            try:
                self._execute_single_browsable_plot_function(func,
                                                             build_static_image=build_static_image)
            except Exception as e:
                msg = f"Error when executing the browsable plot function {name}:{e}."
                self.log_warning(msg)

    def _get_all_ai_inspectable_functions(self) -> dict:
        """
        Get all the AI inspectable functions.

        Returns:
            dict: The AI inspectable functions.
        """
        return dict(
            [(name, func) for name, func in self.get_browser_functions() if
             has_visual_analyze_prompt(func)])

    def get_analyzed_result_prompt(self) -> Union[str, None]:
        """
        Get the natual language description of the analyzed result for AI.

        Returns
        -------
        str: The prompt to analyze the result.
        """
        return None

    def _execute_single_browsable_plot_function(self, func: callable,
                                                build_static_image=False):
        """
        Execute the browsable plot function. The result and image will be stored in the function object
        attributes '_result' and '_image'.

        Parameters:
            func (callable): The browsable plot function.
            build_static_image (bool): Whether to build the static image.

        """
        f_args, f_kwargs = (
            func._browser_function_args,
            func._browser_function_kwargs,
        )

        # For compatibility, select the argument that the function
        # accepts with inspect
        sig = inspect.signature(func)

        # Extract the parameter names that the function accepts
        valid_parameter_names = set(sig.parameters.keys())

        # Filter the kwargs
        filtered_kwargs = {
            k: v for k, v in f_kwargs.items() if k in valid_parameter_names}

        result = None

        try:
            result = func(*f_args, **filtered_kwargs)
            if build_static_image:
                from leeq.utils.ai.utils import matplotlib_plotly_to_pil
                image = matplotlib_plotly_to_pil(result)
                self._browser_function_images[func.__qualname__] = image

        except Exception as e:
            self.log_warning(
                f"Error when executing {func.__qualname__} with parameters ({f_args},{f_kwargs}): {e}"
            )
            self.log_warning(f"Ignore the error and continue.")
            self.log_warning(f"{e}")
            raise e

        self._browser_function_results[func.__qualname__] = result

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

        if inspection_method != 'fitting_only':
            for name, func in self._get_all_ai_inspectable_functions().items():

                if self._ai_inspection_results.get(func.__qualname__) is None:
                    try:
                        self._run_ai_inspection_on_single_function(func)
                    except Exception as e:
                        self.log_warning(
                            f"Error when doing get AI inspection on {func.__qualname__}: {e}"
                        )
                        self.log_warning(f"Ignore the error and continue.")
                        self.log_warning(f"{e}")

            ai_inspection_results = {key.split('.')[-1]: val['analysis'] for key, val
                                     in
                                     self._ai_inspection_results.items()}

        fitting_results = self.get_analyzed_result_prompt()

        if fitting_results is not None and inspection_method != 'visual_only':
            ai_inspection_results['fitting'] = fitting_results

        if self._experiment_result_analysis_instructions is not None:

            if self._ai_final_analysis is None or ignore_cache:
                spinner_id = show_spinner(
                    f"AI is analyzing the experiment results...")

                run_args_prompt = f"""
Document of this experiment:
{self.run.__doc__}
"""
                #Running arguments:
                #{self.retrieve_args(self.run)}

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

    def run_ai_inspection(self):
        for name, func in self.get_browser_functions():
            inspect_answer = self._run_ai_inspection_on_single_function(func)
            if inspect_answer is not None:
                color = 'light_green' if inspect_answer['success'] else 'light_red'
                html = dict_to_html(inspect_answer)

                display_chat(agent_name=f"Inspection AI",
                             content='<br>' + html,
                             background_color=color)

    @classmethod
    def get_browser_functions(cls):
        tag_name = "_browser_function"
        tagged_methods = []
        # iterate through all the methods in the class
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if getattr(method, tag_name, None):
                tagged_methods.append((name, method))
        return tagged_methods

    def _before_run(self):
        """
        Pre run method to be called before the experiment is run.
        """
        pass

    def _post_run(self):
        """
        Post run method to be called after the experiment is run.
        """
        if self.to_run_ai_inspection:
            self.show_plots()

        if self.to_run_ai_inspection:
            self.run_ai_inspection()

    def show_plots(self):
        for name, func in self.get_browser_functions():
            try:
                self._execute_single_browsable_plot_function(func)
                result = self._browser_function_results[func.__qualname__]
            except Exception as e:
                self.log_warning(
                    f"Error when executing the browsable plot function {name}:{e}."
                )
                self.log_warning(f"Ignore the error and continue.")
                self.log_warning(f"{e}")
                continue

            try:
                if isinstance(result, plotly.graph_objs.Figure):
                    result.show()
                if isinstance(result, matplotlib.figure.Figure):
                    from matplotlib import pyplot as plt
                    display(result)
                    plt.close(result)
            except Exception as e:
                self.log_warning(
                    f"Error when displaying experiment result of {func.__qualname__}: {e}"
                )
                self.log_warning(f"Ignore the error and continue.")
                self.log_warning(f"{e}")

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
