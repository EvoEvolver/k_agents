from __future__ import annotations
from typing import TYPE_CHECKING

from k_agents.inspection.vlms import matplotlib_plotly_to_pil, run_visual_inspection
from k_agents.io_interface import show_spinner, hide_spinner, dict_to_html, display_chat

if TYPE_CHECKING:
    from k_agents.experiment import Experiment


class InspectionAgent:

    def __init__(self, label):
        self.label = label

    def run(self, Experiment):
        pass

class TextInspectionAgent(InspectionAgent):

    def __init__(self, func):
        super().__init__(func.__qualname__)
        self.func = func

    def run(self, exp: Experiment):
        res = self.func(exp)
        if isinstance(res, str):
            res = {self.func.__name__: res}
        color = "light_blue"
        if 'success' in res:
            color = 'light_green' if res['success'] else 'light_red'
        html = dict_to_html(res)
        display_chat(agent_name=f"Inspection Agent", content='<br>' + html,
                     background_color=color)
        return res

class VisualInspectionAgent(InspectionAgent):

    def __init__(self, plot_func, prompt, calling_path):
        super().__init__(plot_func.__qualname__)
        self.plot_func = plot_func
        self.prompt = prompt
        self.calling_path = calling_path

    def run(self, exp: Experiment):
        spinner_id = show_spinner(f"Inspection agent reading the plot...")
        figure_obj = self.plot_func(exp)
        image = matplotlib_plotly_to_pil(figure_obj)
        inspect_answer = run_visual_inspection(image, self.prompt, self.calling_path)
        hide_spinner(spinner_id)
        if inspect_answer is not None:
            color = 'light_green' if inspect_answer['success'] else 'light_red'
            html = dict_to_html(inspect_answer)
            display_chat(agent_name=f"Inspection Agent", content='<br>' + html,
                         background_color=color)

        self.save_figure_to_exp(exp, figure_obj, image)
        return inspect_answer

    def save_figure_to_exp(self, exp: Experiment, figure_obj, image):
        exp._plot_function_result_objs[self.label] = figure_obj
        exp._plot_function_images[self.label] = image



