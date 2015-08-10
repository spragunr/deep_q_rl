"""
Author: Nathan Sprague
"""

import numpy as np
import theano
import unittest
import numpy.testing
import lasagne

import deep_q_rl.q_network as q_network

class ChainMDP(object):
    """Simple markov chain style MDP.  Three "rooms" and one absorbing
    state. States are encoded for the q_network as arrays with
    indicator entries. E.g. [1, 0, 0, 0] encodes state 0, and [0, 1,
    0, 0] encodes state 1.  The absorbing state is [0, 0, 0, 1]

    Action 0 moves the agent left, departing the maze if it is in state 0.
    Action 1 moves the agent to the right, departing the maze if it is in
    state 2.

    The agent receives a reward of .7 for departing the chain on the left, and
    a reward of 1.0 for departing the chain on the right.

    Assuming deterministic actions and a discount rate of .5, the
    correct Q-values are:

    .7|.25,  .35|.5, .25|1.0,  0|0
    """

    def __init__(self, success_prob=1.0):
        self.num_actions = 2
        self.num_states = 4
        self.success_prob = success_prob

        self.actions = [np.array([[0]], dtype='int32'),
                        np.array([[1]], dtype='int32')]

        self.reward_zero = np.array([[0]], dtype=theano.config.floatX)
        self.reward_left = np.array([[.7]], dtype=theano.config.floatX)
        self.reward_right = np.array([[1.0]], dtype=theano.config.floatX)

        self.states = []
        for i in range(self.num_states):
            self.states.append(np.zeros((1, 1, 1, self.num_states),
                                        dtype=theano.config.floatX))
            self.states[-1][0, 0, 0, i] = 1

    def act(self, state, action_index):

        """
        action 0 is left, 1 is right.
        """
        state_index =  np.nonzero(state[0, 0, 0, :])[0][0]

        next_index = state_index
        if np.random.random() < self.success_prob:
            next_index = state_index + action_index * 2 - 1

        # Exit left
        if next_index == -1:
            return self.reward_left, self.states[-1], np.array([[True]])

        # Exit right
        if next_index == self.num_states - 1:
            return self.reward_right, self.states[-1], np.array([[True]])

        if np.random.random() < self.success_prob:
            return (self.reward_zero,
                    self.states[state_index + action_index * 2 - 1],
                    np.array([[False]]))
        else:
            return (self.reward_zero, self.states[state_index],
                    np.array([[False]]))


class LinearTests(unittest.TestCase):
    """With no neural network, and simple sgd updates, the deep
    Q-learning code operates as good-ol-fashioned Q-learning.  These
    tests check that the basic updates code is working correctly.
    """
    def setUp(self):

        # Divide the desired learning rate by two, because loss is
        # defined as L^2, not 1/2 L^2.
        self.learning_rate = .1 / 2.0

        self.discount = .5
        self.mdp = ChainMDP()


    def all_q_vals(self, net):
        """ Helper method to get the entire Q-table """

        q_vals = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        for i in range(self.mdp.num_states):
            q_vals[i, :] = net.q_vals(self.mdp.states[i])
        return q_vals

    def train(self, net, steps):
        mdp = self.mdp
        for _ in range(steps):
            state = mdp.states[np.random.randint(0, mdp.num_states-1)]
            action_index = np.random.randint(0, mdp.num_actions)
            reward, next_state, terminal = mdp.act(state, action_index)

            net.train(state, mdp.actions[action_index], reward, next_state,
                      terminal)

    def test_updates_sgd_no_freeze(self):
        freeze_interval = -1
        net = q_network.DeepQLearner(self.mdp.num_states, 1,
                                     self.mdp.num_actions, 1,
                                     self.discount,
                                     self.learning_rate, 0, 0, 0, 0,
                                     freeze_interval, 1, 'linear',
                                     'sgd', 'sum', 1.0)

        mdp = self.mdp

        # Depart left:
        state = mdp.states[0]
        action_index = 0
        reward, next_state, terminal = mdp.act(state, action_index)
        net.train(state, mdp.actions[action_index], reward, next_state,
                  terminal)

        numpy.testing.assert_almost_equal(self.all_q_vals(net),
                                          [[.07, 0], [0, 0], [0, 0], [0, 0]])

        # Depart right:
        state = mdp.states[-2]
        action_index = 1
        reward, next_state, terminal = mdp.act(state, action_index)
        net.train(state, mdp.actions[action_index], reward, next_state,
                  terminal)

        numpy.testing.assert_almost_equal(self.all_q_vals(net),
                                          [[.07, 0], [0, 0], [0, .1], [0, 0]])

        # Move into leftmost state
        state = mdp.states[1]
        action_index = 0
        reward, next_state, terminal = mdp.act(state, action_index)
        net.train(state, mdp.actions[action_index], reward, next_state,
                  terminal)

        numpy.testing.assert_almost_equal(self.all_q_vals(net),
                                          [[.07, 0], [0.0035, 0], [0, .1],
                                           [0, 0]])


    def test_convergence_sgd_no_freeze(self):
        freeze_interval = -1
        net = q_network.DeepQLearner(self.mdp.num_states, 1,
                                     self.mdp.num_actions, 1,
                                     self.discount,
                                     self.learning_rate, 0, 0, 0, 0,
                                     freeze_interval, 1, 'linear',
                                     'sgd', 'sum', 1.0)


        self.train(net, 1000)

        numpy.testing.assert_almost_equal(self.all_q_vals(net),
                                          [[.7, .25], [.35, .5],
                                           [.25, 1.0], [0., 0.]], 3)


    def test_convergence_random_initialization(self):
        """ This test will only pass if terminal states are handled
        correctly. Otherwise the random initialization of the value of the
        terminal state will propagate back.
        """
        freeze_interval = -1
        net = q_network.DeepQLearner(self.mdp.num_states, 1,
                                     self.mdp.num_actions, 1,
                                     self.discount,
                                     self.learning_rate, 0, 0, 0, 0,
                                     freeze_interval, 1, 'linear',
                                     'sgd', 'sum', 1.0)

        # Randomize initial q-values:
        params = lasagne.layers.helper.get_all_param_values(net.l_out)
        rand = np.random.random(params[0].shape)
        rand = numpy.array(rand,  dtype=theano.config.floatX)
        lasagne.layers.helper.set_all_param_values(net.l_out, [rand])

        self.train(net, 1000)

        numpy.testing.assert_almost_equal(self.all_q_vals(net)[0:3,:],
                                          [[.7, .25],
                                           [.35, .5],
                                           [.25, 1.0]], 3)




    def test_convergence_sgd_permanent_freeze(self):
        freeze_interval = 1000000
        net = q_network.DeepQLearner(self.mdp.num_states, 1,
                                     self.mdp.num_actions, 1,
                                     self.discount,
                                     self.learning_rate, 0, 0, 0, 0,
                                     freeze_interval, 1, 'linear',
                                     'sgd', 'sum', 1.0)

        self.train(net, 1000)

        numpy.testing.assert_almost_equal(self.all_q_vals(net),
                                          [[.7, 0], [0, 0],
                                           [0, 1.0], [0., 0.]], 3)

    def test_convergence_sgd_frequent_freeze(self):
        freeze_interval = 2
        net = q_network.DeepQLearner(self.mdp.num_states, 1,
                                     self.mdp.num_actions, 1,
                                     self.discount,
                                     self.learning_rate, 0, 0, 0, 0,
                                     freeze_interval, 1, 'linear',
                                     'sgd', 'sum', 1.0)

        self.train(net, 1000)

        numpy.testing.assert_almost_equal(self.all_q_vals(net),
                                          [[.7, .25], [.35, .5],
                                           [.25, 1.0], [0., 0.]], 3)

    def test_convergence_sgd_one_freeze(self):
        freeze_interval = 500
        net = q_network.DeepQLearner(self.mdp.num_states, 1,
                                     self.mdp.num_actions, 1,
                                     self.discount,
                                     self.learning_rate, 0, 0, 0, 0,
                                     freeze_interval, 1, 'linear',
                                     'sgd', 'sum', 1.0)

        self.train(net, freeze_interval * 2)

        numpy.testing.assert_almost_equal(self.all_q_vals(net),
                                          [[.7, 0], [.35, .5],
                                           [0, 1.0], [0., 0.]], 3)

if __name__ == "__main__":
    unittest.main()
