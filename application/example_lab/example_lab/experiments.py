import numpy as np
from plotly import graph_objects as go
from k_agents.experiment.experiment import Experiment
from k_agents.inspection.decorator import visual_inspection, text_inspection


class GenerateRandomLines(Experiment):

    def run(self, secret_parameter=2):
        """
        Generate a random number of lines.
        """
        self.n_lines = np.random.randint(1, secret_parameter)


    @visual_inspection("The experiment is considered successful if the plot contains 3 lines. Else, suggest increase the secret_parameter by 2.")
    def plot(self):
        fig = go.Figure()
        for i in range(self.n_lines):
            x = np.linspace(0, 10, 100)
            y = np.array([1/self.n_lines * i for _ in range(100)])
            fig.add_trace(go.Scatter(x=x, y=y))
        return fig

class GenerateRandomDots(Experiment):
    def run(self):
        """
        Generate a random number of dots.
        """
        ...

    @text_inspection
    def result(self):
        return "The experiment is successful"