"""
This class is a deep Q-Network as described in:

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

Author: Nathan Sprague
"""
import numpy as np
import layers
import cc_layers
import theano
import theano.tensor as T
import cPickle
import copy

theano.config.exception_verbosity = 'high'

def copy_layers(layer_list):
    """ Perform a shallow copy of a list of network layers.  """
    new_list = [None] * len(layer_list)
    new_list[0] = copy.copy(layer_list[0]) # copy the input layer
    for i in range(1, len(layer_list)):
        new_list[i] = copy.copy(layer_list[i])
        new_list[i].input_layer = new_list[i-1]
    return new_list

def build_mask(matrix, entries, value):
    """
    matrix - m x n tensor
    entries - column tensor with m entries, each containing an integer
              indicating a column number
    value - the value to place into designated columns

    Returns: a theano tensor with the appropriate column entries
             set to value.
    """
    seq = T.arange(matrix.shape[0])

    def set_ones_at_positions(row, mask, entries):
        mask_subtensor = mask[row, entries[row, 0]]
        return T.set_subtensor(mask_subtensor, value)

    result, updates = theano.scan(fn=set_ones_at_positions,
                                  outputs_info=matrix,
                                  sequences=[seq],
                                  non_sequences=entries)
    result = result[-1]
    return result


class CNNQLearner(object):
    """ Reinforcement learning agent implemented using a convolutional
    neural network."""

    def __init__(self, num_actions, phi_length, width, height,
                 discount=.9, learning_rate=.01,
                 batch_size=32,
                 approximator='none'):
        self._batch_size = batch_size
        self._num_input_features = phi_length
        self._phi_length = phi_length
        self._img_width = width
        self._img_height = height
        self._discount = discount
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.scale_input_by = 255.0

        # CONSTRUCT THE LAYERS
        self.q_layers = []
        self.q_layers.append(layers.Input2DLayer(self._batch_size,
                                               self._num_input_features,
                                               self._img_height,
                                               self._img_width,
                                               self.scale_input_by))

        if approximator == 'cuda_conv':
            self.q_layers.append(cc_layers.ShuffleBC01ToC01BLayer(
                    self.q_layers[-1]))
            self.q_layers.append(
                cc_layers.CudaConvnetConv2DLayer(self.q_layers[-1],
                                                 n_filters=16,
                                                 filter_size=8,
                                                 stride=4,
                                                 weights_std=.01,
                                                 init_bias_value=0.1))
            self.q_layers.append(
                cc_layers.CudaConvnetConv2DLayer(self.q_layers[-1],
                                                 n_filters=32,
                                                 filter_size=4,
                                                 stride=2,
                                                 weights_std=.01,
                                                 init_bias_value=0.1))
            self.q_layers.append(cc_layers.ShuffleC01BToBC01Layer(
                    self.q_layers[-1]))

        elif approximator == 'conv':
            self.q_layers.append(layers.StridedConv2DLayer(self.q_layers[-1],
                                                         n_filters=16,
                                                         filter_width=8,
                                                         filter_height=8,
                                                         stride_x=4,
                                                         stride_y=4,
                                                         weights_std=.01,
                                                         init_bias_value=0.01))

            self.q_layers.append(layers.StridedConv2DLayer(self.q_layers[-1],
                                                         n_filters=32,
                                                         filter_width=4,
                                                         filter_height=4,
                                                         stride_x=2,
                                                         stride_y=2,
                                                         weights_std=.01,
                                                         init_bias_value=0.01))
        if approximator == 'cuda_conv' or approximator == 'conv':

            self.q_layers.append(layers.DenseLayer(self.q_layers[-1],
                                                   n_outputs=256,
                                                   weights_std=0.01,
                                                   init_bias_value=0.1,
                                                   dropout=0,
                                                   nonlinearity=layers.rectify))

            self.q_layers.append(
                layers.DenseLayer(self.q_layers[-1],
                                  n_outputs=num_actions,
                                  weights_std=0.01,
                                  init_bias_value=0.1,
                                  dropout=0,
                                  nonlinearity=layers.identity))


        if approximator == 'none':
            self.q_layers.append(\
                layers.DenseLayerNoBias(self.q_layers[-1],
                                        n_outputs=num_actions,
                                        weights_std=0.00,
                                        dropout=0,
                                        nonlinearity=layers.identity))


        self.q_layers.append(layers.OutputLayer(self.q_layers[-1]))

        for i in range(len(self.q_layers)-1):
            print self.q_layers[i].get_output_shape()


        # Now create a network (using the same weights)
        # for next state q values
        self.next_layers = copy_layers(self.q_layers)
        self.next_layers[0] = layers.Input2DLayer(self._batch_size,
                                                  self._num_input_features,
                                                  self._img_width,
                                                  self._img_height,
                                                  self.scale_input_by)
        self.next_layers[1].input_layer = self.next_layers[0]

        self.rewards = T.col()
        self.actions = T.icol()

        # Build the loss function ...
        q_vals = self.q_layers[-1].predictions()
        next_q_vals = self.next_layers[-1].predictions()
        next_maxes = T.max(next_q_vals, axis=1, keepdims=True)
        target = self.rewards + discount * next_maxes
        target = theano.gradient.consider_constant(target)
        diff = target - q_vals
        # Zero out all entries for actions that were not chosen...
        mask = build_mask(T.zeros_like(diff), self.actions, 1.0)
        diff_masked = diff * mask
        error = T.mean(diff_masked ** 2)
        self._loss = error * diff_masked.shape[1] #

        self._parameters = layers.all_parameters(self.q_layers[-1])

        self._idx = T.lscalar('idx')

        # CREATE VARIABLES FOR INPUT AND OUTPUT
        self.states_shared = theano.shared(
            np.zeros((1, 1, 1, 1), dtype=theano.config.floatX))
        self.states_shared_next = theano.shared(
            np.zeros((1, 1, 1, 1), dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
            np.zeros((1, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.actions_shared = theano.shared(
            np.zeros((1, 1), dtype='int32'), broadcastable=(False, True))

        self._givens = \
            {self.q_layers[0].input_var:
             self.states_shared[self._idx*self._batch_size:
                                (self._idx+1)*self._batch_size, :, :, :],
             self.next_layers[0].input_var:
             self.states_shared_next[self._idx*self._batch_size:
                                     (self._idx+1)*self._batch_size, :, :, :],

             self.rewards:
             self.rewards_shared[self._idx*self._batch_size:
                                 (self._idx+1)*self._batch_size, :],
             self.actions:
             self.actions_shared[self._idx*self._batch_size:
                                 (self._idx+1)*self._batch_size, :]
             }

        self._updates = layers.gen_updates_rmsprop_and_nesterov_momentum(\
            self._loss, self._parameters, learning_rate=self.learning_rate,
            rho=0.9, momentum=0.9, epsilon=1e-6)

        self._train = theano.function([self._idx], self._loss,
                                      givens=self._givens,
                                      updates=self._updates)
        self._compute_loss = theano.function([self._idx],
                                             self._loss,
                                             givens=self._givens)
        self._compute_q_vals = \
            theano.function([self.q_layers[0].input_var],
                            self.q_layers[-1].predictions(),
                            on_unused_input='ignore')

        #self.load_weights('network_file_3.pkl')

    def load_weights(self, file_name):
        """
        Load weights for the first two convolution layers
        and the hidden layer from a pickled network.
        """
        net_file = open(file_name, 'r')
        net = cPickle.load(net_file)
        # initial convolution layer
        self.q_layers[2].W.set_value(net.q_layers[2].W.get_value())
        self.q_layers[2].b.set_value(net.q_layers[2].b.get_value())
        # second convolution layer
        self.q_layers[3].W.set_value(net.q_layers[3].W.get_value())
        self.q_layers[3].b.set_value(net.q_layers[3].b.get_value())
        # hidden layer
        self.q_layers[5].b.set_value(net.q_layers[5].b.get_value())
        self.q_layers[5].b.set_value(net.q_layers[5].b.get_value())


    def q_vals(self, state):
        """ Return an array of q-values for the indicated state (phi) 
        """
        state_batch = np.zeros((self._batch_size,
                          self._phi_length,
                          self._img_height,
                          self._img_width), dtype=theano.config.floatX)
        state_batch[0, ...] = state
        return self._compute_q_vals(state_batch)[0, :]

    def choose_action(self, state, epsilon):
        """ 
        Choose a random action with probability epsilon,
        or return the optimal action. 
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions-1)
        else:
            return np.argmax(self.q_vals(state))

    def train(self, states, actions, rewards, next_states,
              terminals, epochs=1):
        num_batches_valid = states.shape[0] // self._batch_size
        self.states_shared.set_value(states)
        self.states_shared_next.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        for epoch in xrange(epochs):
            losses = []
            for b in xrange(num_batches_valid):
                loss = self._train(b)
                losses.append(loss)

            mean_train_loss = np.sqrt(np.mean(losses))
            return mean_train_loss
