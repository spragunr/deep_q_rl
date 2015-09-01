import random
from ale_agent_base import AgentBase


class AgentRandom(AgentBase):
    def __init__(self, params):
        super(AgentRandom, self).__init__(params)
        self.action_set = None

    def initialize(self, action_set):
        self.action_set = action_set

    def start_episode(self, observation):
        return self.step(None, None)

    def step(self, reward, observation):
        return random.randint(0, len(self.action_set) - 1)

    def end_episode(self, reward, terminal):
        pass

    def start_epoch(self, epoch):
        pass

    def finish_epoch(self, epoch):
        pass

    def start_testing(self, epoch):
        pass

    def finish_testing(self, epoch):
        pass
