from abc import ABCMeta, abstractmethod


class QLearner:
    __metaclass__ = ABCMeta

    def __init__(self,
                 num_actions,
                 input_width, input_height, num_frames,
                 parameters):
        pass

    @abstractmethod
    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.
        Arguments:
        states - b x f x h x w numpy array, where b is batch size,
        f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)
        Returns: average loss
        """
        pass

    @abstractmethod
    def q_vals(self, state):
        pass
