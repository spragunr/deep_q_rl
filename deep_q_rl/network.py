import copy
import joblib
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import cuda_convnet
from theano.printing import Print as pp

def rmsprop_nesterov(cost, params, lr=0.001, rho=0.9, momentum=0.7, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        mo_p = theano.shared(p.get_value() * 0.)
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g**2
        step = lr * g / T.sqrt(acc_new + epsilon)
        v = momentum * mo_p - step
        w = p + momentum * v - step
        updates.append((acc, acc_new))
        updates.append((mo_p, v))
        updates.append((p, w))
    return updates

class DeepQLearner:
    def __init__(self, input_width, input_height, output_dim, num_frames, batch_size):
        self.input_width = input_width
        self.input_height = input_height
        self.output_dim = output_dim
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.gamma = 0.99 # discount factor
        self.rho = 0.99
        self.lr = 0.00025 # learning rate
        self.momentum = 0.95
        self.freeze_targets = True

        self.l_out = self.build_network(input_width, input_height, output_dim, num_frames, batch_size)
        if self.freeze_targets:
            self.next_l_out = self.build_network(input_width, input_height, output_dim, num_frames, batch_size)
            self.reset_q_hat()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
#        terminals = T.icol('terminals')

        self.states_shared = theano.shared(np.zeros((batch_size, num_frames, input_height, input_width), dtype=theano.config.floatX))
        self.next_states_shared = theano.shared(np.zeros((batch_size, num_frames, input_height, input_width), dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(np.zeros((batch_size, 1), dtype=theano.config.floatX), broadcastable=(False,True))
        self.actions_shared = theano.shared(np.zeros((batch_size, 1), dtype='int32'), broadcastable=(False,True))
#        self.terminals_shared = theano.shared(np.zeros((batch_size, 1), dtype='int32'), broadcastable=(False,True))

        q_vals = self.l_out.get_output(states / 255.0)
        if self.freeze_targets:
            next_q_vals = self.next_l_out.get_output(next_states / 255.0)
        else:
            next_q_vals = self.l_out.get_output(next_states / 255.0)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = rewards + self.gamma * T.max(next_q_vals, axis=1, keepdims=True)
        diff = target - q_vals[T.arange(batch_size), actions.reshape((-1,))].reshape((-1,1))
        loss = T.mean(diff ** 2)

        params = lasagne.layers.helper.get_all_params(self.l_out)
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
#            terminals: self.terminals_shared
        }
        if self.momentum > 0:
            updates = rmsprop_nesterov(loss, params, self.lr, self.rho, self.momentum, 1e-2)
        else:
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho, 1e-6)
        self._train = theano.function([], [loss, q_vals], updates=updates, givens=givens)
        self._q_vals = theano.function([], q_vals, givens={ states: self.states_shared })

    def build_network(self, input_width, input_height, output_dim, num_frames, batch_size):
        l_in = lasagne.layers.InputLayer(
                shape=(batch_size, num_frames, input_width, input_height)
                )

        l_in = cuda_convnet.bc01_to_c01b(l_in)

        l_conv1 = cuda_convnet.Conv2DCCLayer(
                l_in,
                num_filters=32,
                filter_size=(8,8),
                strides=(4,4),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
                l_conv1,
                num_filters=64,
                filter_size=(4,4),
                strides=(2,2),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )

        l_conv3 = cuda_convnet.Conv2DCCLayer(
                l_conv2,
                num_filters=64,
                filter_size=(3,3),
                strides=(1,1),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )

        l_conv3 = cuda_convnet.c01b_to_bc01(l_conv3)

        l_hidden1 = lasagne.layers.DenseLayer(
                l_conv3,
                num_units=512,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1)
                )

        l_out = lasagne.layers.DenseLayer(
                l_hidden1,
                num_units=output_dim,
                nonlinearity=None,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1)
                )

        return l_out

    def build_small_network(self, input_width, input_height, output_dim, num_frames, batch_size):
        l_in = lasagne.layers.InputLayer(
                shape=(batch_size, num_frames, input_width, input_height)
                )

        l_in = cuda_convnet.bc01_to_c01b(l_in)

        l_conv1 = cuda_convnet.Conv2DCCLayer(
                l_in,
                num_filters=16,
                filter_size=(8,8),
                strides=(4,4),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
                l_conv1,
                num_filters=32,
                filter_size=(4,4),
                strides=(2,2),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )

        l_conv2 = cuda_convnet.c01b_to_bc01(l_conv2)

        l_hidden1 = lasagne.layers.DenseLayer(
                l_conv2,
                num_units=256,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1)
                )

        l_out = lasagne.layers.DenseLayer(
                l_hidden1,
                num_units=output_dim,
                nonlinearity=None,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1)
                )

        return l_out

    def train(self, states, actions, rewards, next_states, terminals):
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
#        self.terminals_shared.set_value(np.logical_not(terminals))
        loss, _ = self._train()
        return np.sqrt(loss)

    def q_vals(self, state):
        states = np.zeros((self.batch_size, self.num_frames, self.input_width, self.input_height), dtype=theano.config.floatX)
        states[0,...] = state.reshape((1, self.num_frames, self.input_width, self.input_height))
        self.states_shared.set_value(states)
        return self._q_vals()[0]

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.output_dim)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        print "reset_q_hat()"
        if self.freeze_targets:
            lasagne.layers.helper.set_all_param_values(self.next_l_out, copy.copy(lasagne.layers.helper.get_all_param_values(self.l_out)))

def main():
  net = DeepQLearner(84, 84, 16, 4, 32)

if __name__ == '__main__':
  main()
