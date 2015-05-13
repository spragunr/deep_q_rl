"""
Layers using the cuda-convnet Theano wrappers that are part of pylearn2.
"""

# Copyright (c) 2014, Sander Dieleman
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the {organization} nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import theano 
import theano.tensor as T
import numpy as np

import layers 

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.sandbox.cuda_convnet.stochastic_pool import StochasticMaxPool, WeightedMaxPool
from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm
from theano.sandbox.cuda.basic_ops import host_from_gpu


class CudaConvnetInput2DLayer(layers.Input2DLayer):
    """
    Like Input2DLayer, but the data is expected to be in c01b order instead of bc01.
    """
    def get_output_shape(self):
        return (self.n_features, self.width, self.height, self.mb_size) # c01b instead of bc01



class CudaConvnetConv2DLayer(object):
    def __init__(self, input_layer, n_filters, filter_size, weights_std, init_bias_value, stride=1, nonlinearity=layers.rectify, dropout=0., partial_sum=None, pad=0, untie_biases=False):
        """
        Only the valid border mode is supported.

        n_filters should be a multiple of 16
        """
        self.input_layer = input_layer
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.stride = stride
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.partial_sum = partial_sum
        self.pad = pad
        self.untie_biases = untie_biases
        # if untie_biases == True, each position in the output map has its own bias (as opposed to having the same bias everywhere for a given filter)
        self.mb_size = self.input_layer.mb_size

        self.input_shape = self.input_layer.get_output_shape()

        self.filter_shape = (self.input_shape[0], filter_size, filter_size, n_filters)

        self.W = layers.shared_single(4) # theano.shared(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)

        if self.untie_biases:
            self.b = layers.shared_single(3)
        else:
            self.b = layers.shared_single(1) # theano.shared(np.ones(n_filters).astype(np.float32) * self.init_bias_value)

        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

        self.filter_acts_op = FilterActs(stride=self.stride, partial_sum=self.partial_sum, pad=self.pad)

    def reset_params(self):
        self.W.set_value(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)

        if self.untie_biases:
            self.b.set_value(np.ones(self.get_output_shape()[:3]).astype(np.float32) * self.init_bias_value)
        else:
            self.b.set_value(np.ones(self.n_filters).astype(np.float32) * self.init_bias_value)

    def get_output_shape(self):
        #output_width = (self.input_shape[1] + 2 * self.pad - self.filter_size + self.stride) // self.stride
        output_width = np.ceil((self.input_shape[1] + 2 * self.pad - self.filter_size + self.stride) / float(self.stride))
        #output_height = (self.input_shape[2] + 2 * self.pad  - self.filter_size + self.stride) // self.stride        
        output_height = np.ceil((self.input_shape[2] + 2 * self.pad  - self.filter_size + self.stride) / float(self.stride))
        output_shape = (self.n_filters, output_width, output_height, self.mb_size)
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
                # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.untie_biases:
            conved += self.b.dimshuffle(0, 1, 2, 'x')
        else:
            conved += self.b.dimshuffle(0, 'x', 'x', 'x')

        return self.nonlinearity(conved)




class CudaConvnetPooling2DLayer(object):
    def __init__(self, input_layer, pool_size, stride=None): # pool_size is an INTEGER here!
        """
        pool_size is an INTEGER, not a tuple. We can only do square pooling windows.
        
        if the stride is none, it is taken to be the same as the pool size.

        borders are never ignored.
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.pool_op = MaxPool(ds=self.pool_size, stride=self.stride)

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape() # convert to list because we cannot assign to a tuple element
        w, h = input_shape[1], input_shape[2]

        new_w = int(np.ceil(float(w - self.pool_size + self.stride) / self.stride))
        new_h = int(np.ceil(float(h - self.pool_size + self.stride) / self.stride))

        return (input_shape[0], new_w, new_h, input_shape[3])

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        contiguous_input = gpu_contiguous(input)
        return self.pool_op(contiguous_input)




class CudaConvnetStochasticPooling2DLayer(object):
    def __init__(self, input_layer, pool_size, stride=None): # pool_size is an INTEGER here!
        """
        This implements stochastic pooling as in Zeiler et al. 2013 to replace max pooling.
        Pooling is stochastic by default. When dropout_active=True, weighted pooling is used
        instead. As a result it is not possible to enable/disable stochastic pooling and
        dropout separately within a network, but the use cases for that should be rare.
        Usually we want both on during training, and both off at test time.

        pool_size is an INTEGER, not a tuple. We can only do square pooling windows.
        
        if the stride is none, it is taken to be the same as the pool size.

        borders are never ignored.
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.stochastic_pool_op = StochasticMaxPool(ds=self.pool_size, stride=self.stride)
        self.weighted_pool_op = WeightedMaxPool(ds=self.pool_size, stride=self.stride)

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape() # convert to list because we cannot assign to a tuple element
        w, h = input_shape[1], input_shape[2]

        new_w = int(np.ceil(float(w - self.pool_size + self.stride) / self.stride))
        new_h = int(np.ceil(float(h - self.pool_size + self.stride) / self.stride))

        return (input_shape[0], new_w, new_h, input_shape[3])

    def output(self, dropout_active=True, *args, **kwargs):
        input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)
        contiguous_input = gpu_contiguous(input)

        if dropout_active:
            return self.stochastic_pool_op(contiguous_input)
        else:
            return self.weighted_pool_op(contiguous_input)






class CudaConvnetCrossMapNormLayer(object):
    def __init__(self, input_layer, alpha=1e-4, beta=0.75, size_f=5, blocked=True):
        self.alpha = alpha
        self.beta = beta
        self.size_f = size_f
        self.blocked = blocked
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.norm_op = CrossMapNorm(size_f=size_f, add_scale=alpha, pow_scale=beta, blocked=blocked)

    def get_output_shape(self):
        # output shape is the same as the input shape
        return self.input_layer.get_output_shape() 

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        contiguous_input = gpu_contiguous(input)
        return self.norm_op(contiguous_input)[0]




class ShuffleC01BToBC01Layer(object):
    """
    This layer dimshuffles 4D input for interoperability between C01B and BC01 ops.
    C01B (cuda convnet) -> BC01 (theano)
    """
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return (input_shape[3], input_shape[0], input_shape[1], input_shape[2])

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return input.dimshuffle(3, 0, 1, 2)


class ShuffleBC01ToC01BLayer(object):
    """
    This layer dimshuffles 4D input for interoperability between C01B and BC01 ops.
    BC01 (theano) -> C01B (cuda convnet)
    """
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return input.dimshuffle(1, 2, 3, 0)




class CudaConvnetCircularConv2DLayer(object):
    def __init__(self, input_layer, n_filters, filter_size, weights_std, init_bias_value, stride=1, nonlinearity=layers.rectify, dropout=0., partial_sum=None, untie_biases=False):
        """
        This is a convolution which is circular in the 0-direction, and valid in the 1-direction.

        n_filters should be a multiple of 16
        """
        self.input_layer = input_layer
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.stride = stride
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.partial_sum = partial_sum
        self.untie_biases = untie_biases
        # if untie_biases == True, each position in the output map has its own bias (as opposed to having the same bias everywhere for a given filter)
        self.mb_size = self.input_layer.mb_size

        self.input_shape = self.input_layer.get_output_shape()

        self.filter_shape = (self.input_shape[0], filter_size, filter_size, n_filters)

        self.W = layers.shared_single(4) # theano.shared(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)

        if self.untie_biases:
            self.b = layers.shared_single(3)
        else:
            self.b = layers.shared_single(1) # theano.shared(np.ones(n_filters).astype(np.float32) * self.init_bias_value)

        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

        self.filter_acts_op = FilterActs(stride=self.stride, partial_sum=self.partial_sum)

    def reset_params(self):
        self.W.set_value(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)

        if self.untie_biases:
            self.b.set_value(np.ones(self.get_output_shape()[:3]).astype(np.float32) * self.init_bias_value)
        else:
            self.b.set_value(np.ones(self.n_filters).astype(np.float32) * self.init_bias_value)

    def get_output_shape(self):
        # output_width = (self.input_shape[1] - self.filter_size + self.stride) // self.stride
        output_width = self.input_shape[1] // self.stride # because it's a circular convolution, this dimension is just divided by the stride.
        output_height = (self.input_shape[2] - self.filter_size + self.stride) // self.stride # in this direction it's still valid though.       
        output_shape = (self.n_filters, output_width, output_height, self.mb_size)
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
                # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.
            input = input / retain_prob * mask

        # pad input so the valid convolution amounts to a circular one.
        # we need to copy (filter_size - stride) values from one side to the other
        input_padded = T.zeros((input.shape[0], input.shape[1] + self.filter_size - self.stride, input.shape[2], input.shape[3]))
        input_padded = T.set_subtensor(input_padded[:, :input.shape[1], :, :], input)
        input_padded = T.set_subtensor(input_padded[:, input.shape[1]:, :, :], input[:, :self.filter_size - self.stride, :, :])

        contiguous_input = gpu_contiguous(input_padded)
        contiguous_filters = gpu_contiguous(self.W)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.untie_biases:
            conved += self.b.dimshuffle(0, 1, 2, 'x')
        else:
            conved += self.b.dimshuffle(0, 'x', 'x', 'x')

        return self.nonlinearity(conved)




def shuffle_pool_unshuffle(input_layer, *args, **kwargs):
    """
    The Krizhevskhy max pooling layer only supports square input. This function provides
    a workaround that uses Theano's own max pooling op, flanked by two shuffling operations:
    c01b to bc01 before pooling, and bc01 to c01b afterwards.
    """
    l_bc01 = ShuffleC01BToBC01Layer(input_layer)
    l_pool = layers.Pooling2DLayer(l_bc01, *args, **kwargs)
    l_c01b = ShuffleBC01ToC01BLayer(l_pool)

    return l_c01b




class StochasticPoolingC01BLayer(object):
    """
    Stochastic pooling implemented in Theano using reshapes, since the Pylearn2 class for it is
    way too slow.

    This only works for c01b, i.e. it assumes that the dimensions to pool over are (1, 2).
    It's also required that the dimensions are a multiple of the pool size (no incomplete pools).

    epsilon is used to prevent division by 0, it is added to all probabilities,
    so that when all activations are 0, the distribution is uniform.
    """
    def __init__(self, input_layer, pool_size, epsilon=1e-12):
        """
        pool_size: the number of inputs to be pooled together.
        """
        self.pool_size = pool_size
        self.epsilon = epsilon
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.mb_size = self.input_layer.mb_size

        self.params = []
        self.bias_params = []

    def get_output_shape(self):
        output_shape = list(self.input_shape) # make a mutable copy
        output_shape[1] = output_shape[1] // self.pool_size
        output_shape[2] = output_shape[2] // self.pool_size
        return tuple(output_shape)

    def output(self, dropout_active=True, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)

        output_shape = self.get_output_shape()
        pool_shape = (output_shape[0], output_shape[1], self.pool_size, output_shape[2], self.pool_size, output_shape[3])
        merged_shape = (output_shape[0], output_shape[1], output_shape[2], output_shape[3], self.pool_size**2)
        flat_shape = (output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3], self.pool_size**2)
        input_reshaped = input.reshape(pool_shape).transpose(0, 1, 3, 5, 2, 4).reshape(flat_shape) #pools are now in axis 4

        input_reshaped += self.epsilon # add a small constant to prevent division by 0 in what follows.

        if dropout_active:
            probabilities = input_reshaped / input_reshaped.sum(axis=1, keepdims=True)
            samples = layers.srng.multinomial(pvals=probabilities, dtype=theano.config.floatX)
            output_flat = T.sum(input_reshaped * samples, axis=1)
            output = output_flat.reshape(output_shape)
        else:
            # no dropout, so compute the weighted average instead.
            # this amounts to the sum of squares normalised by the sum of the values.
            numerator = T.sum(input_reshaped**2, axis=1)
            denominator = T.sum(input_reshaped, axis=1)
            output_flat = numerator / denominator
            output = output_flat.reshape(output_shape)
            
        return output
