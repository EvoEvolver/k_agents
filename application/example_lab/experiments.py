import numpy as np
from plotly import graph_objects as go
from k_agents.experiment.experiment import Experiment
from k_agents.inspection.decorator import visual_inspection


class GenerateRandomWave(Experiment):

    def run(self):
        # Generate a random wave
        n_waves = np.random.randint(1, 6)
        # Generate a random sinusoidal wave with n_waves
        self.x = np.linspace(0, 1, 100)
        period = 1/n_waves
        self.result = np.sin(self.x*2*np.pi/period)

    @visual_inspection("The experiment is considered successful if the plot contains 3 complete sinusoidal waves.")
    def plot(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x, y=self.result))
        return fig