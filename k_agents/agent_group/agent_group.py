from __future__ import annotations
import uuid

from typing import List

import numpy as np

from mllm import get_embeddings
from mllm.utils.maps import p_map

from .recall_logger import RecallLogger, to_log_item
from .w_memory import WMemoryItem, WMemorySuppressingItem, WorkingMemory
#from ..translation.bm25_retrieval import get_bm25_score


class RetrievableAgent:
    """
    Long term memory items
    """

    def __init__(self, description):
        self.description = description
        self.agent_id = uuid.uuid4().int
        self.data = None

    def get_score(self, w_memory: WorkingMemory):
        pass

    def run_agent(self, w_memory: WorkingMemory) -> AgentResult:
        return AgentResult(self, True)

    def __hash__(self):
        return self.agent_id

    def __eq__(self, other):
        return self.agent_id == other.agent_id

    def __repr__(self):
        return "RetrievableAgent('" + self.description + "')"


def get_bm25_score_for_agent(agent: EmbedAgent, w_memory: WorkingMemory):
    if len(agent._embed_src) == 0:
        return 0
    bm25_scores = get_bm25_score(agent._embed_src, w_memory.stimuli)
    bm25_score = max(bm25_scores)
    return bm25_score


class EmbedAgent(RetrievableAgent):
    def __init__(self, description, embed_src: List[str]):
        super().__init__(description)
        self._embed_src = []
        self.embeddings = None
        self.add_embed_src(embed_src)

    def add_embed_src(self, src: str | List[str]):
        if isinstance(src, str):
            src = [src]
        for s in src:
            if len(s) != 0:
                self._embed_src.append(s)
        self.embeddings = get_embeddings(self._embed_src)

    def get_score(self, w_memory: WorkingMemory):
        if len(self.embeddings) == 0:
            return 0
        similarities = np.dot(np.array(self.embeddings), w_memory.stimuli_embeddings.T)
        similarities = np.max(similarities, axis=0)
        score = np.max(similarities)
        #bm25_score = get_bm25_score_for_agent(self, w_memory) - 0.7
        #score = max(bm25_score, score)
        return score


    def run_agent(self, w_memory: WorkingMemory) -> AgentResult:
        raise NotImplementedError


class AgentResult:
    """
    Container for results generated by an agent run.

    :ivar success: True if the run was successful, False otherwise
    :ivar agent: the agent that generated this result
    :ivar new_wm_items: a list of working memory items generated by the run
    :ivar tags_to_remove: a list of working memory item tags to be removed
    """
    success: bool
    agent: RetrievableAgent
    new_wm_items: List[WMemoryItem]
    tags_to_remove: List[str]

    def __init__(self, agent, success):
        self.success = success
        self.agent = agent
        self.new_wm_items = []
        self.tags_to_remove = []
        self.function_to_apply = []

    def add_new_wm_content(self, content: str, tag: str):
        self.new_wm_items.append(WMemoryItem(content, tag))

    def add_new_wm_item(self, item: WMemoryItem):
        self.new_wm_items.append(item)

    def add_suppressing_wm_item(self, lifetime=3, tag=None):
        """
        Add an item suppressing the agent for a certain amount of time
        """
        if isinstance(lifetime, list):
            # sample from the list
            lifetime = np.random.choice(lifetime)
        self.add_new_wm_item(WMemorySuppressingItem(self.agent, lifetime=lifetime))


class AgentGroupResult:
    """
    Container for results generated by a long term memory recall.

    :ivar agent_results: a list of agent results generated in the recall
    """
    agent_results: List[AgentResult]

    def __init__(self, agent_results: List[AgentResult]):
        self.agent_results = agent_results

    def update_wm_from_res(self, wm: WorkingMemory):
        # remove tags
        tags_to_remove = set()
        for agent_res in self.agent_results:
            tags_to_remove.update(agent_res.tags_to_remove)
        wm.remove_item_by_tags(tags_to_remove)

        # add new items
        for agent_res in self.agent_results:
            for item in agent_res.new_wm_items:
                wm.add_item(item)

        for agent_res in self.agent_results:
            for func in agent_res.function_to_apply:
                func(wm)

        # refresh cache
        wm.refresh_cache()

    def no_success_agent(self):
        for agent_res in self.agent_results:
            if agent_res.success:
                return False
        return True


class AgentGroup:
    def __init__(self):
        self.agents: List[RetrievableAgent] = []
        self.src_embeddings = None
        self.recall_workers = 8

    @classmethod
    def from_agents(cls, agents: List[RetrievableAgent]):
        agent_group = cls()
        agent_group.agents = agents
        return agent_group

    def add_agent(self, agent: RetrievableAgent):
        self.agents.append(agent)

    def get_scores(self, w_memory: WorkingMemory):
        if not self.agents:
            raise ValueError("RetrievableAgent contains no agents")
        agent_scores = []
        for agent in self.agents:
            agent_scores.append(agent.get_score(w_memory))
        return np.array(agent_scores)

    def get_agents_by_score(self, w_memory: WorkingMemory, k=5,
                           excluded_agent: List[RetrievableAgent] = None) -> List[RetrievableAgent]:
        scores = self.get_scores(w_memory)
        if scores is None:
            return []
        highest_k_indices = np.argsort(scores)
        if excluded_agent is None:
            excluded_agent = []
        must_trigger_agent = []
        agent_list = []
        for i in range(len(highest_k_indices) - 1, -1, -1):
            this_index = highest_k_indices[i]
            if len(agent_list) >= k:
                break
            this_agent = self.agents[this_index]
            if this_agent in excluded_agent:
                continue
            # don't add the agent if its score is lower than 0
            this_score = scores[this_index]
            if this_score < 0:
                continue
            #if this_score > 1.0:
            #    must_trigger_agent.append(this_agent)
            else:
                agent_list.append(this_agent)
        agent_list.extend(must_trigger_agent)
        return agent_list


    #@standard_multi_attempts
    def recall_by_wm(self, w_memory: WorkingMemory, top_k=5) -> AgentGroupResult:
        """
        Recall agents from working memory.
        """
        if len(self.agents) == 0:
            return AgentGroupResult([])

        excluded_agent = w_memory.get_suppressed_agents()
        agents_from_score = self.get_agents_by_score(w_memory, top_k, excluded_agent)
        agents_from_score = list(set(agents_from_score))

        if len(RecallLogger.active_loggers) > 0:
            old_stimuli = w_memory.stimuli
            old_wm_in_prompt = w_memory.get_in_prompt_format()

        def run_agent(agent: RetrievableAgent):
            return agent.run_agent(w_memory)

        triggered_agents = []
        new_wm_items = []
        agent_results = []

        for agent, res in p_map(run_agent, agents_from_score, title="Recalling"):
            res: AgentResult
            if res.success:
                agent_results.append(res)
                triggered_agents.append(res.agent)
                new_wm_items.extend(res.new_wm_items)

        if len(RecallLogger.active_loggers) > 0:
            log = f"""
                    {to_log_item(agents_from_score, "RetrievableAgent has run")}
                    {to_log_item(triggered_agents, "Triggered agents")}
                    {to_log_item(old_wm_in_prompt, "Old Working Memory")}
                    {to_log_item(old_stimuli, "Stimuli")}
                    """
            #{to_log_item(new_wm_items, "New Working Memory items")}
            RecallLogger.add_log_to_all(log)

        return AgentGroupResult(agent_results)
