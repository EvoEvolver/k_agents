from __future__ import annotations

from typing import TYPE_CHECKING

from k_agents.utils import Singleton

if TYPE_CHECKING:
    from k_agents.translation.agent import TranslationAgentGroup
    from k_agents.variable_table import VariableTable


class TranslationAgentEnv(Singleton):

    def __init__(self):
        if not self._initialized:
            self.translation_agents: TranslationAgentGroup = None
            self.translation_var_table: VariableTable = None
        super().__init__()
