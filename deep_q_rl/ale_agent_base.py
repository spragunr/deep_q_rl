from abc import ABCMeta, abstractmethod


class AgentBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, parameters):
        pass

    @abstractmethod
    def initialize(self, action_set):
        pass

    @abstractmethod
    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.
        Arguments:
        observation - height x width numpy array
        Returns:
        An integer action
        """
        pass

    @abstractmethod
    def step(self, reward, observation):
        """
        This method is called each time step.
        Arguments:
        reward      - Real valued reward.
        observation - A height x width numpy array
        Returns:
        An integer action.
        """
        pass

    @abstractmethod
    def end_episode(self, reward, terminal):
        """
        This function is called once at the end of an episode.
        Arguments:
        reward      - Real valued reward.
        terminal    - Whether the episode ended intrinsically
        (ie we didn't run out of steps)
        Returns:
        None
        """
        pass

    @abstractmethod
    def start_epoch(self, epoch):
        pass

    @abstractmethod
    def finish_epoch(self, epoch):
        pass

    @abstractmethod
    def start_testing(self, epoch):
        pass

    @abstractmethod
    def finish_testing(self, epoch):
        pass