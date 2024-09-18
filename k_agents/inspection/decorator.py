import inspect
from k_agents.inspection.agents import TextInspectionAgent, VisualInspectionAgent


def text_inspection(func):

    func._inspection_agent = TextInspectionAgent(func)

    return func


def visual_inspection(prompt: str):
    """
    Decorator function for the functions that used to visualize data of the class.
    It is used to register the prompt to analyze the data.

    Parameters:
        prompt (str): The prompt to be registered.

    Returns:
        Any: The return value of the function.
    """
    calling_path = inspect.stack()[1].filename

    def inner_func(plot_func):
        """
        Decorator function for the functions that used to visualize data of the class.
        It is used to register the prompt to analyze the data.
        The function must be a method of a LoggableObject.

        Parameters:
            plot_func (function): The function to be registered.

        Returns:
            Any: The same function.
        """
        plot_func._browser_function = True
        plot_func._is_plot_function = True
        plot_func._inspection_agent = VisualInspectionAgent(plot_func, prompt, calling_path)
        return plot_func

    return inner_func
