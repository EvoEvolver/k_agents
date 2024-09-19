from k_agents.translation.agent import TranslationAgent
from k_agents.utils import Singleton
from k_agents.variable_table import VariableTable


class TranslationAgentEnv(Singleton):

    def __init__(self):
        if not self._initialized:
            self.translation_agent: TranslationAgent = None
            self.translation_var_table: VariableTable = None
        super().__init__()