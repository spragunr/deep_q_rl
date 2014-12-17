
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

# Some Modifications by Nathan Sprague 8/14


import numpy as np
import theano.tensor as T
import theano
from theano.tensor.signal.conv import conv2d as sconv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
import os
import cPickle as pickle


srng = RandomStreams()

# nonlinearities

sigmoid = T.nnet.sigmoid

tanh = T.tanh

def rectify(x):
    return T.maximum(x, 0.0)
    
def identity(x):
    # To create a linear layer.
    return x

def compress(x, C=10000.0):
    return T.log(1 + C * x ** 2) # no binning matrix here of course

def compress_abs(x, C=10000.0):
    return T.log(1 + C * abs(x))


def all_layers(layer):
    """
    Recursive function to gather all layers below the given layer (including the given layer)
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return [layer]
    elif isinstance(layer, ConcatenateLayer):
        return sum([all_layers(i) for i in layer.input_layers], [layer])
    else:
        return [layer] + all_layers(layer.input_layer)

def all_parameters(layer):
    """
    Recursive function to gather all parameters, starting from the output layer
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return []
    elif isinstance(layer, ConcatenateLayer):
        return sum([all_parameters(i) for i in layer.input_layers], [])
    else:
        return layer.params + all_parameters(layer.input_layer)

def all_bias_parameters(layer):
    """
    Recursive function to gather all bias parameters, starting from the output layer
    """    
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return []
    elif isinstance(layer, ConcatenateLayer):
        return sum([all_bias_parameters(i) for i in layer.input_layers], [])
    else:
        return layer.bias_params + all_bias_parameters(layer.input_layer)

def all_non_bias_parameters(layer):
    return [p for p in all_parameters(layer) if p not in all_bias_parameters(layer)]


def gather_rescaling_updates(layer, c):
    """
    Recursive function to gather weight rescaling updates when the constant is the same for all layers.
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return []
    elif isinstance(layer, ConcatenateLayer):
        return sum([gather_rescaling_updates(i, c) for i in layer.input_layers], [])
    else:
        if hasattr(layer, 'rescaling_updates'):
            updates = layer.rescaling_updates(c)
        else:
            updates = []
        return updates + gather_rescaling_updates(layer.input_layer, c)



def get_param_values(layer):
    params = all_parameters(layer)
    return [p.get_value() for p in params]


def set_param_values(layer, param_values):
    params = all_parameters(layer)
    for p, pv in zip(params, param_values):
        p.set_value(pv)


def reset_all_params(layer):
    for l in all_layers(layer):
        if hasattr(l, 'reset_params'):
            l.reset_params()


    

def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum, weight_decay):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        v = momentum * mparam_i - weight_decay * learning_rate * param_i  - learning_rate * grad_i
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))
    return updates


# using the alternative formulation of nesterov momentum described at https://github.com/lisa-lab/pylearn2/pull/136
# such that the gradient can be evaluated at the current parameters.

def gen_updates_nesterov_momentum(loss, all_parameters, learning_rate, momentum, weight_decay):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))
    return updates


def gen_updates_nesterov_momentum_no_bias_decay(loss, all_parameters, all_bias_parameters, learning_rate, momentum, weight_decay):
    """
    Nesterov momentum, but excluding the biases from the weight decay.
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        if param_i in all_bias_parameters:
            full_grad = grad_i
        else:
            full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))
    return updates


gen_updates = gen_updates_nesterov_momentum


def gen_updates_sgd(loss, all_parameters, learning_rate):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    return updates



def gen_updates_adagrad(loss, all_parameters, learning_rate=1.0, epsilon=1e-6):
    """
    epsilon is not included in the typical formula, 

    See "Notes on AdaGrad" by Chris Dyer for more info.
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    all_accumulators = [theano.shared(param.get_value()*0.) for param in all_parameters] # initialise to zeroes with the right shape

    updates = []
    for param_i, grad_i, acc_i in zip(all_parameters, all_grads, all_accumulators):
        acc_i_new = acc_i + grad_i**2
        updates.append((acc_i, acc_i_new))
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc_i_new + epsilon)))

    return updates


# Added by NRS
# These were  helpful:
# http://climin.readthedocs.org/en/latest/rmsprop.html
# https://github.com/lisa-lab/pylearn2/pull/136
def gen_updates_rmsprop_and_nesterov_momentum(loss, all_parameters, 
                                              learning_rate,
                                              rho=0.9, momentum=0.9, 
                                              epsilon=1e-6):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        rms_i = theano.shared(param_i.get_value()*0.)

        rms_i_new = rho * rms_i + (1 - rho) * grad_i**2

        step = learning_rate *  grad_i /  T.sqrt(rms_i_new + epsilon)

        v = momentum * mparam_i - step # new momemtum
        w = param_i + momentum * v - step # new parameter values

        updates.append((rms_i, rms_i_new ))
        updates.append((mparam_i, v))
        updates.append((param_i, w))
    return updates



def gen_updates_rmsprop(loss, all_parameters, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    epsilon is not included in Hinton's video, but to prevent problems with relus repeatedly having 0 gradients, it is included here.

    Watch this video for more info: http://www.youtube.com/watch?v=O3sxAc4hxZU (formula at 5:20)

    also check http://climin.readthedocs.org/en/latest/rmsprop.html
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    all_accumulators = [theano.shared(param.get_value()*0.) for param in all_parameters] # initialise to zeroes with the right shape
    # all_accumulators = [theano.shared(param.get_value()*1.) for param in all_parameters] # initialise with 1s to damp initial gradients

    updates = []
    for param_i, grad_i, acc_i in zip(all_parameters, all_grads, all_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i**2
        updates.append((acc_i, acc_i_new))
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc_i_new + epsilon)))

    return updates


def gen_updates_adadelta(loss, all_parameters, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    """
    in the paper, no learning rate is considered (so learning_rate=1.0). Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does not become 0).

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to work for multiple datasets (MNIST, speech).

    see "Adadelta: an adaptive learning rate method" by Matthew Zeiler for more info.
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    all_accumulators = [theano.shared(param.get_value()*0.) for param in all_parameters] # initialise to zeroes with the right shape
    all_delta_accumulators = [theano.shared(param.get_value()*0.) for param in all_parameters]

    # all_accumulators: accumulate gradient magnitudes
    # all_delta_accumulators: accumulate update magnitudes (recursive!)

    updates = []
    for param_i, grad_i, acc_i, acc_delta_i in zip(all_parameters, all_grads, all_accumulators, all_delta_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i**2
        updates.append((acc_i, acc_i_new))

        update_i = grad_i * T.sqrt(acc_delta_i + epsilon) / T.sqrt(acc_i_new + epsilon) # use the 'old' acc_delta here
        updates.append((param_i, param_i - learning_rate * update_i))

        acc_delta_i_new = rho * acc_delta_i + (1 - rho) * update_i**2
        updates.append((acc_delta_i, acc_delta_i_new))

    return updates    



def shared_single(dim=2):
    """
    Shortcut to create an undefined single precision Theano shared variable.
    """
    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype='float32'))



class InputLayer(object):
    def __init__(self, mb_size, n_features, length):
        self.mb_size = mb_size
        self.n_features = n_features
        self.length = length
        self.input_var = T.tensor3('input')

    def get_output_shape(self):
        return (self.mb_size, self.n_features, self.length)

    def output(self, *args, **kwargs):
        """
        return theano variable
        """
        return self.input_var


class FlatInputLayer(InputLayer):
    def __init__(self, mb_size, n_features):
        self.mb_size = mb_size
        self.n_features = n_features
        self.input_var = T.matrix('input')

    def get_output_shape(self):
        return (self.mb_size, self.n_features)

    def output(self, *args, **kwargs):
        """
        return theano variable
        """
        return self.input_var


# NRS - add scale argument
class Input2DLayer(object):
    def __init__(self, mb_size, n_features, width, height, scale=1.0):
        self.mb_size = mb_size
        self.n_features = n_features
        self.width = width
        self.height = height
        self.input_var = T.tensor4('input')
        self.scale = scale

    def get_output_shape(self):
        return (self.mb_size, self.n_features, self.width, self.height)

    def output(self, *args, **kwargs):
        return self.input_var / self.scale




class PoolingLayer(object):
    def __init__(self, input_layer, ds_factor, ignore_border=False):
        self.ds_factor = ds_factor
        self.input_layer = input_layer
        self.ignore_border = ignore_border
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        output_shape = list(self.input_layer.get_output_shape()) # convert to list because we cannot assign to a tuple element
        if self.ignore_border:
            output_shape[-1] = int(np.floor(float(output_shape[-1]) / self.ds_factor))
        else:
            output_shape[-1] = int(np.ceil(float(output_shape[-1]) / self.ds_factor))
        return tuple(output_shape)

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return max_pool_2d(input, (1, self.ds_factor), self.ignore_border)



class Pooling2DLayer(object):
    def __init__(self, input_layer, pool_size, ignore_border=False): # pool_size is a tuple
        self.pool_size = pool_size # a tuple
        self.input_layer = input_layer
        self.ignore_border = ignore_border
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        output_shape = list(self.input_layer.get_output_shape()) # convert to list because we cannot assign to a tuple element
        if self.ignore_border:
            output_shape[-2] = int(np.floor(float(output_shape[-2]) / self.pool_size[0]))
            output_shape[-1] = int(np.floor(float(output_shape[-1]) / self.pool_size[1]))
        else:
            output_shape[-2] = int(np.ceil(float(output_shape[-2]) / self.pool_size[0]))
            output_shape[-1] = int(np.ceil(float(output_shape[-1]) / self.pool_size[1]))
        return tuple(output_shape)

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return max_pool_2d(input, self.pool_size, self.ignore_border)



class GlobalPooling2DLayer(object):
    """
    Global pooling across the entire feature map, useful in NINs.
    """
    def __init__(self, input_layer, pooling_function='mean'):
        self.input_layer = input_layer
        self.pooling_function = pooling_function
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        return self.input_layer.get_output_shape()[:2] # this effectively removes the last 2 dimensions

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        if self.pooling_function == 'mean':
            out = input.mean([2, 3])
        elif self.pooling_function == 'max':
            out = input.max([2, 3])
        elif self.pooling_function == 'l2':
            out = T.sqrt((input ** 2).mean([2, 3]))

        return out



class DenseLayer(object):
    def __init__(self, input_layer, n_outputs, weights_std, init_bias_value, nonlinearity=rectify, dropout=0.):
        self.n_outputs = n_outputs
        self.input_layer = input_layer
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.mb_size = self.input_layer.mb_size

        input_shape = self.input_layer.get_output_shape()
        self.n_inputs = int(np.prod(input_shape[1:]))
        self.flatinput_shape = (self.mb_size, self.n_inputs)

        self.W = shared_single(2) # theano.shared(np.random.randn(self.n_inputs, n_outputs).astype(np.float32) * weights_std)
        self.b = shared_single(1) # theano.shared(np.ones(n_outputs).astype(np.float32) * self.init_bias_value)
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(np.random.randn(self.n_inputs, self.n_outputs).astype(np.float32) * self.weights_std)
        self.b.set_value(np.ones(self.n_outputs).astype(np.float32) * self.init_bias_value)

    def get_output_shape(self):
        return (self.mb_size, self.n_outputs)

    def output(self, input=None, dropout_active=True, *args, **kwargs): # use the 'dropout_active' keyword argument to disable it at test time. It is on by default.
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)
        if len(self.input_layer.get_output_shape()) > 2:
            input = input.reshape(self.flatinput_shape)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            input = input / retain_prob * srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.

        return self.nonlinearity(T.dot(input, self.W) + self.b.dimshuffle('x', 0))

    def rescaled_weights(self, c): # c is the maximal norm of the weight vector going into a single filter.
        norms = T.sqrt(T.sqr(self.W).mean(0, keepdims=True))
        scale_factors = T.minimum(c / norms, 1)
        return self.W * scale_factors

    def rescaling_updates(self, c):
        return [(self.W, self.rescaled_weights(c))]



#added by nrs 6/23/14
class DenseLayerNoBias(object):
    def __init__(self, input_layer, n_outputs, weights_std, nonlinearity=rectify, dropout=0.):
        self.n_outputs = n_outputs
        self.input_layer = input_layer
        self.weights_std = np.float32(weights_std)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.mb_size = self.input_layer.mb_size

        input_shape = self.input_layer.get_output_shape()
        self.n_inputs = int(np.prod(input_shape[1:]))
        self.flatinput_shape = (self.mb_size, self.n_inputs)

        self.W = shared_single(2) # theano.shared(np.random.randn(self.n_inputs, n_outputs).astype(np.float32) * weights_std)
        self.params = [self.W]
        self.bias_params = []
        self.reset_params()

    def reset_params(self):
        self.W.set_value(np.random.randn(self.n_inputs, self.n_outputs).astype(np.float32) * self.weights_std)

    def get_output_shape(self):
        return (self.mb_size, self.n_outputs)

    def output(self, input=None, dropout_active=True, *args, **kwargs): # use the 'dropout_active' keyword argument to disable it at test time. It is on by default.
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)
        if len(self.input_layer.get_output_shape()) > 2:
            input = input.reshape(self.flatinput_shape)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            input = input / retain_prob * srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.

        return self.nonlinearity(T.dot(input, self.W))

    def rescaled_weights(self, c): # c is the maximal norm of the weight vector going into a single filter.
        norms = T.sqrt(T.sqr(self.W).mean(0, keepdims=True))
        scale_factors = T.minimum(c / norms, 1)
        return self.W * scale_factors

    def rescaling_updates(self, c):
        return [(self.W, self.rescaled_weights(c))]





class ConvLayer(object):
    def __init__(self, input_layer, n_filters, filter_length, weights_std, init_bias_value, nonlinearity=rectify, flip_conv_dims=False, dropout=0.):
        self.n_filters = n_filters
        self.filter_length = filter_length
        self.stride = 1
        self.input_layer = input_layer
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.nonlinearity = nonlinearity
        self.flip_conv_dims = flip_conv_dims
        self.dropout = dropout
        self.mb_size = self.input_layer.mb_size

        self.input_shape = self.input_layer.get_output_shape()
        ' MB_size, N_filters, Filter_length '

#        if len(self.input_shape) == 2:
#            self.filter_shape = (n_filters, 1, filter_length)
#        elif len(self.input_shape) == 3:
#            self.filter_shape = (n_filters, self.input_shape[1], filter_length)
#        else:
#            raise

        self.filter_shape = (n_filters, self.input_shape[1], filter_length)

        self.W = shared_single(3) # theano.shared(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        self.b = shared_single(1) # theano.shared(np.ones(n_filters).astype(np.float32) * self.init_bias_value)
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        self.b.set_value(np.ones(self.n_filters).astype(np.float32) * self.init_bias_value)

    def get_output_shape(self):
        output_length = (self.input_shape[2] - self.filter_length + self.stride) / self.stride # integer division
        output_shape = (self.input_shape[0], self.n_filters, output_length)
        return output_shape

    def output(self, input=None, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(*args, **kwargs)

        if self.flip_conv_dims: # flip the conv dims to get a faster convolution when the filter_height is 1.
            flipped_input_shape = (self.input_shape[1], self.input_shape[0], self.input_shape[2])
            flipped_input = input.dimshuffle(1, 0, 2)
            conved = sconv2d(flipped_input, self.W, subsample=(1, self.stride), image_shape=flipped_input_shape, filter_shape=self.filter_shape)
            conved = T.addbroadcast(conved, 0) # else dimshuffle complains about dropping a non-broadcastable dimension
            conved = conved.dimshuffle(2, 1, 3)
        else:
            conved = sconv2d(input, self.W, subsample=(1, self.stride), image_shape=self.input_shape, filter_shape=self.filter_shape)
            conved = conved.dimshuffle(0, 1, 3) # gets rid of the obsolete filter height dimension

        return self.nonlinearity(conved + self.b.dimshuffle('x', 0, 'x'))

    # def dropoutput_train(self):
    #     p = self.dropout
    #     input = self.input_layer.dropoutput_train()
    #     if p > 0.:
    #         srng = RandomStreams()
    #         input = input * srng.binomial(self.input_layer.get_output_shape(), p=1 - p, dtype='int32').astype('float32')
    #     return self.output(input)

    # def dropoutput_predict(self):
    #     p = self.dropout
    #     input = self.input_layer.dropoutput_predict()
    #     if p > 0.:
    #         input = input * (1 - p)
    #     return self.output(input)




class StridedConvLayer(object):
    def __init__(self, input_layer, n_filters, filter_length, stride, weights_std, init_bias_value, nonlinearity=rectify, dropout=0.):
        if filter_length % stride != 0:
            print 'ERROR: the filter_length should be a multiple of the stride '
            raise
        if stride == 1:
            print 'ERROR: use the normal ConvLayer instead (stride=1) '
            raise

        self.n_filters = n_filters
        self.filter_length = filter_length
        self.stride = 1
        self.input_layer = input_layer
        self.stride = stride
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.mb_size = self.input_layer.mb_size

        self.input_shape = self.input_layer.get_output_shape()
        ' MB_size, N_filters, Filter_length '


        self.filter_shape = (n_filters, self.input_shape[1], filter_length)

        self.W = shared_single(3) # theano.shared(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        self.b = shared_single(1) # theano.shared(np.ones(n_filters).astype(np.float32) * self.init_bias_value)
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        self.b.set_value(np.ones(self.n_filters).astype(np.float32) * self.init_bias_value)

    def get_output_shape(self):
        output_length = (self.input_shape[2] - self.filter_length + self.stride) / self.stride # integer division
        output_shape = (self.input_shape[0], self.n_filters, output_length)
        return output_shape

    def output(self, input=None, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(*args, **kwargs)
        input_shape = list(self.input_shape) # make a mutable copy

        # if the input is not a multiple of the stride, cut off the end
        if input_shape[2] % self.stride != 0:
            input_shape[2] = self.stride * (input_shape[2] / self.stride)
            input_truncated = input[:, :, :input_shape[2]] # integer division
        else:
            input_truncated = input

        r_input_shape = (input_shape[0], input_shape[1], input_shape[2] / self.stride, self.stride) # (mb size, #out, length/stride, stride)
        r_input = input_truncated.reshape(r_input_shape)

        if self.stride == self.filter_length:
            print " better use a tensordot"
            # r_input = r_input.dimshuffle(0, 2, 1, 3) # (mb size, length/stride, #out, stride)
            conved = T.tensordot(r_input, self.W, np.asarray([[1, 3], [1, 2]]))
            conved = conved.dimshuffle(0, 2, 1)
        elif self.stride == self.filter_length / 2:
            print " better use two tensordots"
            # define separate shapes for the even and odd parts, as they may differ depending on whether the sequence length
            # is an even or an odd multiple of the stride.
            length_even = input_shape[2] // self.filter_length
            length_odd = (input_shape[2] - self.stride) // self.filter_length

            r2_input_shape_even = (input_shape[0], input_shape[1], length_even, self.filter_length)
            r2_input_shape_odd = (input_shape[0], input_shape[1], length_odd, self.filter_length)

            r2_input_even = input[:, :, :length_even * self.filter_length].reshape(r2_input_shape_even)
            r2_input_odd = input[:, :, self.stride:length_odd * self.filter_length + self.stride].reshape(r2_input_shape_odd)

            conved_even = T.tensordot(r2_input_even, self.W, np.asarray([[1,3], [1, 2]]))
            conved_odd = T.tensordot(r2_input_odd, self.W, np.asarray([[1, 3], [1, 2]]))

            conved_even = conved_even.dimshuffle(0, 2, 1)
            conved_odd = conved_odd.dimshuffle(0, 2, 1)

            conved = T.zeros((conved_even.shape[0], conved_even.shape[1], conved_even.shape[2] + conved_odd.shape[2]))

            conved = T.set_subtensor(conved[:, :, ::2], conved_even)
            conved = T.set_subtensor(conved[:, :, 1::2], conved_odd)

        else:
            " use a convolution"
            r_filter_shape = (self.filter_shape[0], self.filter_shape[1], self.filter_shape[2] / self.stride, self.stride)

            r_W = self.W.reshape(r_filter_shape)

            conved = conv2d(r_input, r_W, image_shape=r_input_shape, filter_shape=r_filter_shape)
            conved = conved[:, :, :, 0] # get rid of the obsolete 'stride' dimension

        return self.nonlinearity(conved + self.b.dimshuffle('x', 0, 'x'))

    # def dropoutput_train(self):
    #     p = self.dropout
    #     input = self.input_layer.dropoutput_train()
    #     if p > 0.:
    #         srng = RandomStreams()
    #         input = input * srng.binomial(self.input_layer.get_output_shape(), p=1 - p, dtype='int32').astype('float32')
    #     return self.output(input)

    # def dropoutput_predict(self):
    #     p = self.dropout
    #     input = self.input_layer.dropoutput_predict()
    #     if p > 0.:
    #         input = input * (1 - p)
    #     return self.output(input)



class Conv2DLayer(object):
    def __init__(self, input_layer, n_filters, filter_width, filter_height, weights_std, init_bias_value, nonlinearity=rectify, dropout=0., dropout_tied=False, border_mode='valid'):
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.input_layer = input_layer
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.dropout_tied = dropout_tied  # if this is on, the same dropout mask is applied across the entire input map
        self.border_mode = border_mode
        self.mb_size = self.input_layer.mb_size

        self.input_shape = self.input_layer.get_output_shape()
        ' mb_size, n_filters, filter_width, filter_height '

        self.filter_shape = (n_filters, self.input_shape[1], filter_width, filter_height)

        self.W = shared_single(4) # theano.shared(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        self.b = shared_single(1) # theano.shared(np.ones(n_filters).astype(np.float32) * self.init_bias_value)
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        self.b.set_value(np.ones(self.n_filters).astype(np.float32) * self.init_bias_value)

    def get_output_shape(self):
        if self.border_mode == 'valid':
            output_width = self.input_shape[2] - self.filter_width + 1
            output_height = self.input_shape[3] - self.filter_height + 1
        elif self.border_mode == 'full':
            output_width = self.input_shape[2] + self.filter_width - 1
            output_height = self.input_shape[3] + self.filter_width - 1
        elif self.border_mode == 'same':
            output_width = self.input_shape[2]
            output_height = self.input_shape[3]
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        output_shape = (self.input_shape[0], self.n_filters, output_width, output_height)
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            if self.dropout_tied:
                # tying of the dropout masks across the entire feature maps, so broadcast across the feature maps.
                mask = srng.binomial((input.shape[0], input.shape[1]), p=retain_prob, dtype='int32').astype('float32').dimshuffle(0, 1, 'x', 'x')
            else:
                mask = srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
                # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.
            input = input / retain_prob * mask

        if self.border_mode in ['valid', 'full']:
            conved = conv2d(input, self.W, subsample=(1, 1), image_shape=self.input_shape, filter_shape=self.filter_shape, border_mode=self.border_mode)
        elif self.border_mode == 'same':
            conved = conv2d(input, self.W, subsample=(1, 1), image_shape=self.input_shape, filter_shape=self.filter_shape, border_mode='full')
            shift_x = (self.filter_width - 1) // 2
            shift_y = (self.filter_height - 1) // 2
            conved = conved[:, :, shift_x:self.input_shape[2] + shift_x, shift_y:self.input_shape[3] + shift_y]
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)
        return self.nonlinearity(conved + self.b.dimshuffle('x', 0, 'x', 'x'))

    def rescaled_weights(self, c): # c is the maximal norm of the weight vector going into a single filter.
        weights_shape = self.W.shape
        W_flat = self.W.reshape((weights_shape[0], T.prod(weights_shape[1:])))
        norms = T.sqrt(T.sqr(W_flat).mean(1))
        scale_factors = T.minimum(c / norms, 1)
        return self.W * scale_factors.dimshuffle(0, 'x', 'x', 'x')

    def rescaling_updates(self, c):
        return [(self.W, self.rescaled_weights(c))]




class MaxoutLayer(object):
    def __init__(self, input_layer, n_filters_per_unit, dropout=0.):
        self.n_filters_per_unit = n_filters_per_unit
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.dropout = dropout
        self.mb_size = self.input_layer.mb_size

        self.params = []
        self.bias_params = []

    def get_output_shape(self):
        return (self.input_shape[0], self.input_shape[1] / self.n_filters_per_unit, self.input_shape[2])

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            input = input / retain_prob * srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.

        output = input.reshape((self.input_shape[0], self.input_shape[1] / self.n_filters_per_unit, self.n_filters_per_unit, self.input_shape[2]))
        output = T.max(output, 2)
        return output





class NIN2DLayer(object):
    def __init__(self, input_layer, n_outputs, weights_std, init_bias_value, nonlinearity=rectify, dropout=0., dropout_tied=False):
        self.n_outputs = n_outputs
        self.input_layer = input_layer
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.dropout_tied = dropout_tied # if this is on, the same dropout mask is applied to all instances of the layer across the map.
        self.mb_size = self.input_layer.mb_size

        self.input_shape = self.input_layer.get_output_shape()
        self.n_inputs = self.input_shape[1]

        self.W = shared_single(2) # theano.shared(np.random.randn(self.n_inputs, n_outputs).astype(np.float32) * weights_std)
        self.b = shared_single(1) # theano.shared(np.ones(n_outputs).astype(np.float32) * self.init_bias_value)
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(np.random.randn(self.n_inputs, self.n_outputs).astype(np.float32) * self.weights_std)
        self.b.set_value(np.ones(self.n_outputs).astype(np.float32) * self.init_bias_value)

    def get_output_shape(self):
        return (self.mb_size, self.n_outputs, self.input_shape[2], self.input_shape[3])

    def output(self, input=None, dropout_active=True, *args, **kwargs): # use the 'dropout_active' keyword argument to disable it at test time. It is on by default.
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)
        
        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            if self.dropout_tied:
                # tying of the dropout masks across the entire feature maps, so broadcast across the feature maps.

                 mask = srng.binomial((input.shape[0], input.shape[1]), p=retain_prob, dtype='int32').astype('float32').dimshuffle(0, 1, 'x', 'x')
            else:
                mask = srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
                # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.
            input = input / retain_prob * mask

        prod = T.tensordot(input, self.W, [[1], [0]]) # this has shape (batch_size, width, height, out_maps)
        prod = prod.dimshuffle(0, 3, 1, 2) # move the feature maps to the 1st axis, where they were in the input
        return self.nonlinearity(prod + self.b.dimshuffle('x', 0, 'x', 'x'))






class FilterPoolingLayer(object):
    """
    pools filter outputs from the previous layer. If the pooling function is 'max', the result is maxout.
    supported pooling function:
        - 'max': maxout (max pooling)
        - 'ss': sum of squares (L2 pooling)
        - 'rss': root of the sum of the squares (L2 pooling)
    """
    def __init__(self, input_layer, n_filters_per_unit, dropout=0., pooling_function='max'):
        self.n_filters_per_unit = n_filters_per_unit
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.dropout = dropout
        self.pooling_function = pooling_function
        self.mb_size = self.input_layer.mb_size

        self.params = []
        self.bias_params = []

    def get_output_shape(self):
        return (self.input_shape[0], self.input_shape[1] / self.n_filters_per_unit, self.input_shape[2])

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            input = input / retain_prob * srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.

        output = input.reshape((self.input_shape[0], self.input_shape[1] / self.n_filters_per_unit, self.n_filters_per_unit, self.input_shape[2]))

        if self.pooling_function == "max":
            output = T.max(output, 2)
        elif self.pooling_function == "ss":
            output = T.mean(output**2, 2)
        elif self.pooling_function == "rss":
            # a stabilising constant to prevent NaN in the gradient
            padding = 0.000001
            output = T.sqrt(T.mean(output**2, 2) + padding)
        else:
            raise "Unknown pooling function: %s" % self.pooling_function

        return output




class OutputLayer(object):
    def __init__(self, input_layer, error_measure='mse'):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.error_measure = error_measure
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix() # variable for the labels
        if error_measure == 'maha':
            self.target_cov_var = T.tensor3()

    def error(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)

        # never actually dropout anything on the output layer, just pass it along!

        if self.error_measure == 'mse':
            error = T.mean((input - self.target_var) ** 2)
        elif self.error_measure == 'ce': # cross entropy
            error = T.mean(T.nnet.binary_crossentropy(input, self.target_var))
        elif self.error_measure == 'nca':
            epsilon = 1e-8
            #dist_ij = - T.dot(input, input.T)
            # dist_ij = input
            dist_ij = T.sum((input.dimshuffle(0, 'x', 1) - input.dimshuffle('x', 0, 1)) ** 2, axis=2)
            p_ij_unnormalised = T.exp(-dist_ij) + epsilon
            p_ij_unnormalised = p_ij_unnormalised * (1 - T.eye(self.mb_size)) # set the diagonal to 0
            p_ij = p_ij_unnormalised / T.sum(p_ij_unnormalised, axis=1)
            return - T.mean(p_ij * self.target_var)

            # 
            # p_ij = p_ij_unnormalised / T.sum(p_ij_unnormalised, axis=1)
            # return np.mean(p_ij * self.target_var)
        elif self.error_measure == 'maha':
            # e = T.shape_padright(input - self.target_var)
            # e = (input - self.target_var).dimshuffle((0, 'x', 1))
            # error = T.sum(T.sum(self.target_cov_var * e, 2) ** 2) / self.mb_size

            e = (input - self.target_var)
            eTe = e.dimshuffle((0, 'x', 1)) * e.dimshuffle((0, 1, 'x'))
            error = T.sum(self.target_cov_var * eTe) / self.mb_size
        else:
            1 / 0

        return error

    def error_rate(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        error_rate = T.mean(T.neq(input > 0.5, self.target_var))
        return error_rate

    def predictions(self, *args, **kwargs):
        return self.input_layer.output(*args, **kwargs)




class FlattenLayer(object):
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        size = int(np.prod(input_shape[1:]))
        return (self.mb_size, size)

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return input.reshape(self.get_output_shape())




class ConcatenateLayer(object):
    def __init__(self, input_layers):
        self.input_layers = input_layers
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layers[0].mb_size

    def get_output_shape(self):
        sizes = [i.get_output_shape()[1] for i in self.input_layers] # this assumes the layers are already flat!
        return (self.mb_size, sum(sizes))

    def output(self, *args, **kwargs):
        inputs = [i.output(*args, **kwargs) for i in self.input_layers]
        return T.concatenate(inputs, axis=1)



class ResponseNormalisationLayer(object):
    def __init__(self, input_layer, n, k, alpha, beta):
        """
        n: window size
        k: bias
        alpha: scaling
        beta: power
        """
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        return self.input_layer.get_output_shape()

    def output(self, *args, **kwargs):
        """
        Code is based on https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py
        """
        input = self.input_layer.output(*args, **kwargs)

        half = self.n // 2
        sq = T.sqr(input)
        b, ch, r, c = input.shape
        extra_channels = T.alloc(0., b, ch + 2*half, r, c)
        sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)
        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[:,i:i+ch,:,:]

        scale = scale ** self.beta

        return input / scale




class StridedConv2DLayer(object):
    def __init__(self, input_layer, n_filters, filter_width, filter_height, stride_x, stride_y, weights_std, init_bias_value, nonlinearity=rectify, dropout=0., dropout_tied=False, implementation='convolution'):
        """
        implementation can be:
            - convolution: use conv2d with the subsample parameter
            - unstrided: use conv2d + reshaping so the result is a convolution with strides (1, 1)
            - single_dot: use a large tensor product
            - many_dots: use a bunch of tensor products
        """
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.input_layer = input_layer
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.dropout_tied = dropout_tied  # if this is on, the same dropout mask is applied across the entire input map
        self.implementation = implementation # this controls whether the convolution is computed using theano's op,
        # as a bunch of tensor products, or a single stacked tensor product.
        self.mb_size = self.input_layer.mb_size

        self.input_shape = self.input_layer.get_output_shape()
        ' mb_size, n_filters, filter_width, filter_height '

        self.filter_shape = (n_filters, self.input_shape[1], filter_width, filter_height)

        if self.filter_width % self.stride_x != 0:
            raise RuntimeError("Filter width is not a multiple of the stride in the X direction")

        if self.filter_height % self.stride_y != 0:
            raise RuntimeError("Filter height is not a multiple of the stride in the Y direction")

        self.W = shared_single(4) # theano.shared(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        self.b = shared_single(1) # theano.shared(np.ones(n_filters).astype(np.float32) * self.init_bias_value)
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)  
        self.b.set_value(np.ones(self.n_filters).astype(np.float32) * self.init_bias_value)      

    def get_output_shape(self):
        output_width = (self.input_shape[2] - self.filter_width + self.stride_x) // self.stride_x # integer division
        output_height = (self.input_shape[3] - self.filter_height + self.stride_y) // self.stride_y # integer division
        output_shape = (self.input_shape[0], self.n_filters, output_width, output_height)
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input == None:
            input = self.input_layer.output(dropout_active=dropout_active, *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            if self.dropout_tied:
                # tying of the dropout masks across the entire feature maps, so broadcast across the feature maps.
                mask = srng.binomial((input.shape[0], input.shape[1]), p=retain_prob, dtype='int32').astype('float32').dimshuffle(0, 1, 'x', 'x')
            else:
                mask = srng.binomial(input.shape, p=retain_prob, dtype='int32').astype('float32')
                # apply the input mask and rescale the input accordingly. By doing this it's no longer necessary to rescale the weights at test time.
            input = input / retain_prob * mask

        output_shape = self.get_output_shape()
        W_flipped = self.W[:, :, ::-1, ::-1]

        # crazy convolution stuff!
        if self.implementation == 'single_dot':
            # one stacked product
            num_steps_x = self.filter_width // self.stride_x
            num_steps_y = self.filter_height // self.stride_y
            # print "DEBUG: %d x %d yields %d subtensors" % (num_steps_x, num_steps_y, num_steps_x * num_steps_y)

            # pad the input so all the shifted dot products fit inside. shape is (b, c, w, h)
            # padded_width =  int(np.ceil(self.input_shape[2] / float(self.filter_width))) * self.filter_width # INCORRECT
            # padded_height = int(np.ceil(self.input_shape[3] / float(self.filter_height))) * self.filter_height # INCORRECT

            padded_width =  (self.input_shape[2] // self.filter_width) * self.filter_width + (num_steps_x - 1) * self.stride_x
            padded_height = (self.input_shape[3] // self.filter_height) * self.filter_height + (num_steps_y - 1) * self.stride_y

            # print "DEBUG - PADDED WIDTH: %d" % padded_width
            # print "DEBUG - PADDED HEIGHT: %d" % padded_height

            # at this point, it is possible that the padded_width and height are SMALLER than the input size.
            # so then we have to truncate first.
            truncated_width = min(self.input_shape[2], padded_width)
            truncated_height = min(self.input_shape[3], padded_height)
            input_truncated = input[:, :, :truncated_width, :truncated_height]

            input_padded_shape = (self.input_shape[0], self.input_shape[1], padded_width, padded_height)
            input_padded = T.zeros(input_padded_shape)
            input_padded = T.set_subtensor(input_padded[:, :, :truncated_width, :truncated_height], input_truncated)


            inputs_x = []
            for num_x in xrange(num_steps_x):
                inputs_y = []
                for num_y in xrange(num_steps_y):
                    shift_x = num_x * self.stride_x # pixel shift in the x direction
                    shift_y = num_y * self.stride_y # pixel shift in the y direction

                    width = (input_padded_shape[2] - shift_x) // self.filter_width
                    height = (input_padded_shape[3] - shift_y) // self.filter_height

                    r_input_shape = (input_padded_shape[0], input_padded_shape[1], width, self.filter_width, height, self.filter_height)

                    r_input = input_padded[:, :, shift_x:width * self.filter_width + shift_x, shift_y:height * self.filter_height + shift_y]
                    r_input = r_input.reshape(r_input_shape)

                    inputs_y.append(r_input)

                inputs_x.append(T.stack(*inputs_y))

            inputs_stacked = T.stack(*inputs_x) # shape is (n_x, n_y, b, c, w_x, f_x, w_y, f_y)
            r_conved = T.tensordot(inputs_stacked, W_flipped, np.asarray([[3, 5, 7], [1, 2, 3]]))
            # resulting shape is (n_x, n_y, b, w_x, w_y, n_filters)
            # output needs to be (b, n_filters, w_x * n_x, w_y * n_y)
            r_conved = r_conved.dimshuffle(2, 5, 3, 0, 4, 1) # (b, n_filters, w_x, n_x, w_y, n_y)
            conved = r_conved.reshape((r_conved.shape[0], r_conved.shape[1], r_conved.shape[2] * r_conved.shape[3], r_conved.shape[4] * r_conved.shape[5]))
            # result is (b, n_f, w, h)

            # remove padding
            conved = conved[:, :, :output_shape[2], :output_shape[3]]

            # raise NotImplementedError("single stacked product not implemented yet")
        elif self.implementation == 'many_dots':
            # separate products
            num_steps_x = self.filter_width // self.stride_x
            num_steps_y = self.filter_height // self.stride_y
            # print "DEBUG: %d x %d yields %d subtensors" % (num_steps_x, num_steps_y, num_steps_x * num_steps_y)

            conved = T.zeros(output_shape)

            for num_x in xrange(num_steps_x):
                for num_y in xrange(num_steps_y):
                    shift_x = num_x * self.stride_x # pixel shift in the x direction
                    shift_y = num_y * self.stride_y # pixel shift in the y direction

                    width = (self.input_shape[2] - shift_x) // self.filter_width
                    height = (self.input_shape[3] - shift_y) // self.filter_height

                    if (width == 0) or (height == 0): # we can safely skip this product, it doesn't contribute to the final convolution.
                        # print "DEBUG: WARNING: skipping %d,%d" % (num_x, num_y)
                        continue

                    r_input_shape = (self.input_shape[0], self.input_shape[1], width, self.filter_width, height, self.filter_height)

                    r_input = input[:, :, shift_x:width * self.filter_width + shift_x, shift_y:height * self.filter_height + shift_y]
                    r_input = r_input.reshape(r_input_shape)

                    r_conved = T.tensordot(r_input, W_flipped, np.asarray([[1, 3, 5], [1, 2, 3]])) # shape (b,  w, h, n_filters)
                    r_conved = r_conved.dimshuffle(0, 3, 1, 2) # (b, n_filters, w, h)
                    conved = T.set_subtensor(conved[:, :, num_x::num_steps_x, num_y::num_steps_y], r_conved)

        elif self.implementation == 'unstrided':
            num_steps_x = self.filter_width // self.stride_x
            num_steps_y = self.filter_height // self.stride_y

            # input sizes need to be multiples of the strides, truncate to correct sizes.
            truncated_width =  (self.input_shape[2] // self.stride_x) * self.stride_x
            truncated_height = (self.input_shape[3] // self.stride_y) * self.stride_y
            input_truncated = input[:, :, :truncated_width, :truncated_height]

            r_input_shape = (self.input_shape[0], self.input_shape[1], truncated_width // self.stride_x, self.stride_x, truncated_height // self.stride_y, self.stride_y)
            r_input = input_truncated.reshape(r_input_shape)

            # fold strides into the feature maps dimension
            r_input_folded_shape = (self.input_shape[0], self.input_shape[1] * self.stride_x * self.stride_y, truncated_width // self.stride_x, truncated_height // self.stride_y)
            r_input_folded = r_input.transpose(0, 1, 3, 5, 2, 4).reshape(r_input_folded_shape)

            r_filter_shape = (self.filter_shape[0], self.filter_shape[1], num_steps_x, self.stride_x, num_steps_y, self.stride_y)
            r_W_flipped = W_flipped.reshape(r_filter_shape) # need to operate on the flipped W here, else things get hairy.

            # fold strides into the feature maps dimension
            r_filter_folded_shape = (self.filter_shape[0], self.filter_shape[1] * self.stride_x * self.stride_y, num_steps_x, num_steps_y)
            r_W_flipped_folded = r_W_flipped.transpose(0, 1, 3, 5, 2, 4).reshape(r_filter_folded_shape)
            r_W_folded = r_W_flipped_folded[:, :, ::-1, ::-1] # unflip

            conved = conv2d(r_input_folded, r_W_folded, subsample=(1, 1), image_shape=r_input_folded_shape, filter_shape=r_filter_folded_shape)
            # 'conved' should already have the right shape

        elif self.implementation == 'convolution':
            conved = conv2d(input, self.W, subsample=(self.stride_x, self.stride_y), image_shape=self.input_shape, filter_shape=self.filter_shape)
            # raise NotImplementedError("strided convolution using the theano op not implemented yet")
        else:
            raise RuntimeError("Invalid implementation string: '%s'" % self.implementation)

        return self.nonlinearity(conved + self.b.dimshuffle('x', 0, 'x', 'x'))


    def rescaled_weights(self, c): # c is the maximal norm of the weight vector going into a single filter.
        weights_shape = self.W.shape
        W_flat = self.W.reshape((weights_shape[0], T.prod(weights_shape[1:])))
        norms = T.sqrt(T.sqr(W_flat).mean(1))
        scale_factors = T.minimum(c / norms, 1)
        return self.W * scale_factors.dimshuffle(0, 'x', 'x', 'x')


    def rescaling_updates(self, c):
        return [(self.W, self.rescaled_weights(c))]




class Rot90SliceLayer(object):
    """
    This layer cuts 4 square-shaped parts of out of the input, rotates them 0, 90, 180 and 270 degrees respectively
    so they all have the same orientation, and then stacks them in the minibatch dimension.

    This allows for the same filters to be used in 4 directions.

    IMPORTANT: this increases the minibatch size for all subsequent layers!
    """
    def __init__(self, input_layer, part_size):
        self.input_layer = input_layer
        self.part_size = part_size
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size * 4 # 4 times bigger because of the stacking!

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return (self.mb_size, input_shape[1], self.part_size, self.part_size)

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)

        ps = self.part_size # shortcut 
        part0 = input[:, :, :ps, :ps] # 0 degrees
        part1 = input[:, :, :ps, :-ps-1:-1].dimshuffle(0, 1, 3, 2) # 90 degrees
        part2 = input[:, :, :-ps-1:-1, :-ps-1:-1] # 180 degrees
        part3 = input[:, :, :-ps-1:-1, :ps].dimshuffle(0, 1, 3, 2) # 270 degrees

        return T.concatenate([part0, part1, part2, part3], axis=0)


class Rot90MergeLayer(FlattenLayer):
    """
    This layer merges featuremaps that were separated by the Rot90SliceLayer and flattens them in one go.
    """
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size // 4 # divide by 4 again (it was multiplied by 4 by the Rot90SliceLayer)

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        size = int(np.prod(input_shape[1:])) * 4
        return (self.mb_size, size)

    def output(self, *args, **kwargs):
        input_shape = self.input_layer.get_output_shape()
        input = self.input_layer.output(*args, **kwargs)
        input_r = input.reshape((4, self.mb_size, input_shape[1] * input_shape[2] * input_shape[3])) # split out the 4* dimension
        return input_r.transpose(1, 0, 2).reshape(self.get_output_shape())




class MultiRotSliceLayer(ConcatenateLayer):
    """
    This layer cuts 4 square-shaped parts of out of the input, rotates them 0, 90, 180 and 270 degrees respectively
    so they all have the same orientation, and then stacks them in the minibatch dimension.

    It takes multiple input layers (expected to be multiple rotations of the same image) and stacks the results.
    All inputs should have the same shape!

    This allows for the same filters to be used in many different directions.

    IMPORTANT: this increases the minibatch size for all subsequent layers!

    enabling include_flip also includes flipped versions of all the parts. This doubles the number of views.
    """
    def __init__(self, input_layers, part_size, include_flip=False):
        self.input_layers = input_layers
        self.part_size = part_size
        self.include_flip = include_flip
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layers[0].mb_size * 4 * len(self.input_layers)
        # 4 * num_layers times bigger because of the stacking!
        
        if self.include_flip:
            self.mb_size *= 2 # include_flip doubles the number of views.
        

    def get_output_shape(self):
        input_shape = self.input_layers[0].get_output_shape()
        return (self.mb_size, input_shape[1], self.part_size, self.part_size)

    def output(self, *args, **kwargs):
        parts = []
        for input_layer in self.input_layers:
            input = input_layer.output(*args, **kwargs)
            ps = self.part_size # shortcut 

            if self.include_flip:
                input_representations = [input, input[:, :, :, ::-1]] # regular and flipped
            else:
                input_representations = [input] # just regular

            for input_rep in input_representations:
                part0 = input_rep[:, :, :ps, :ps] # 0 degrees
                part1 = input_rep[:, :, :ps, :-ps-1:-1].dimshuffle(0, 1, 3, 2) # 90 degrees
                part2 = input_rep[:, :, :-ps-1:-1, :-ps-1:-1] # 180 degrees
                part3 = input_rep[:, :, :-ps-1:-1, :ps].dimshuffle(0, 1, 3, 2) # 270 degrees
                parts.extend([part0, part1, part2, part3])

        return T.concatenate(parts, axis=0)


class MultiRotMergeLayer(FlattenLayer):
    """
    This layer merges featuremaps that were separated by the MultiRotSliceLayer and flattens them in one go.
    """
    def __init__(self, input_layer, num_views):
        """
        num_views is the number of different input representations that were merged.
        """
        self.input_layer = input_layer
        self.num_views = num_views
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size // (4 * self.num_views) # divide by total number of parts

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        size = int(np.prod(input_shape[1:])) * (4 * self.num_views)
        return (self.mb_size, size)

    def output(self, *args, **kwargs):
        input_shape = self.input_layer.get_output_shape()
        input = self.input_layer.output(*args, **kwargs)
        input_r = input.reshape((4 * self.num_views, self.mb_size, int(np.prod(input_shape[1:])))) # split out the 4* dimension
        return input_r.transpose(1, 0, 2).reshape(self.get_output_shape())




def sparse_initialisation(n_inputs, n_outputs, sparsity=0.05, std=0.01):
    """
    sparsity: fraction of the weights to each output unit that should be nonzero
    """
    weights = np.zeros((n_inputs, n_outputs), dtype='float32')
    size = int(sparsity * n_inputs)
    for k in xrange(n_outputs):
        indices = np.arange(n_inputs)
        np.random.shuffle(indices)
        indices = indices[:size]
        values = np.random.randn(size).astype(np.float32) * std
        weights[indices, k] = values

    return weights



class FeatureMaxPoolingLayer_old(object):
    """
    OLD implementation using T.maximum iteratively. This turns out to be very slow.

    Max pooling across feature maps. This can be used to implement maxout.
    This is similar to the FilterPoolingLayer, but this version uses a different
    implementation that supports input of any dimensionality and can do pooling 
    across any of the dimensions. It also supports overlapping pooling (the stride
    and downsample factor can be set separately).

    based on code from pylearn2's Maxout implementation.
    https://github.com/lisa-lab/pylearn2/blob/a2b616a384b9f39fa6f3e8d9e316b3af1274e687/pylearn2/models/maxout.py

    IMPORTANT: this layer requires that num_output_features = (feature_dim_size - pool_size + stride) / stride is INTEGER.
    if it isn't, it probably won't work properly.
    """
    def __init__(self, input_layer, pool_size, stride=None, feature_dim=1):
        """
        pool_size: the number of inputs to be pooled together.

        stride: the stride between pools, if not set it defaults to pool_size
        (no overlap)

        feature_dim: the dimension of the input to pool across. By default this is 1
        for both dense and convolutional layers (bc01).
        For c01b, this has to be set to 0.
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.feature_dim = feature_dim
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.mb_size = self.input_layer.mb_size

        self.params = []
        self.bias_params = []

    def get_output_shape(self):
        feature_dim_size = self.input_shape[self.feature_dim]
        out_feature_dim_size = (feature_dim_size - self.pool_size + self.stride) // self.stride
        output_shape = list(self.input_shape) # make a mutable copy
        output_shape[self.feature_dim] = out_feature_dim_size
        return tuple(output_shape)

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)

        indices = [slice(None)] * input.ndim # select everything

        output = None
        for k in xrange(self.pool_size):
            indices[self.feature_dim] = slice(k, None, self.stride) # narrow down the selection for the feature dim
            m = input[tuple(indices)]
            if output is None:
                output = m
            else:
                output = T.maximum(output, m)

        return output



class FeatureMaxPoolingLayer(object):
    """
    Max pooling across feature maps. This can be used to implement maxout.
    This is similar to the FilterPoolingLayer, but this version uses a different
    implementation that supports input of any dimensionality and can do pooling 
    across any of the dimensions.

    IMPORTANT: this layer requires that feature_dim_size is a multiple of pool_size.
    """
    def __init__(self, input_layer, pool_size, feature_dim=1, implementation='max_pool'):
        """
        pool_size: the number of inputs to be pooled together.

        feature_dim: the dimension of the input to pool across. By default this is 1
        for both dense and convolutional layers (bc01).
        For c01b, this has to be set to 0.

        implementation:
            - 'max_pool': uses theano's max_pool_2d - doesn't work for input dimension > 1024!
            - 'reshape': reshapes the tensor to create a 'pool' dimension and then uses T.max.
        """
        self.pool_size = pool_size
        self.feature_dim = feature_dim
        self.implementation = implementation
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.mb_size = self.input_layer.mb_size

        if self.input_shape[self.feature_dim] % self.pool_size != 0:
            raise "Feature dimension is not a multiple of the pool size. Doesn't work!"

        self.params = []
        self.bias_params = []

    def get_output_shape(self):
        feature_dim_size = self.input_shape[self.feature_dim]
        out_feature_dim_size = feature_dim_size // self.pool_size
        output_shape = list(self.input_shape) # make a mutable copy
        output_shape[self.feature_dim] = out_feature_dim_size
        return tuple(output_shape)

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)

        if self.implementation == 'max_pool':
            # max_pool_2d operates on the last 2 dimensions of the input. So shift the feature dim to be last.
            shuffle_order = range(0, self.feature_dim) + range(self.feature_dim + 1, input.ndim) + [self.feature_dim]
            unshuffle_order = range(0, self.feature_dim) + [input.ndim - 1] + range(self.feature_dim, input.ndim - 1)

            input_shuffled = input.dimshuffle(*shuffle_order)
            output_shuffled = max_pool_2d(input_shuffled, (1, self.pool_size))
            output = output_shuffled.dimshuffle(*unshuffle_order)

        elif self.implementation == 'reshape':
            out_feature_dim_size = self.get_output_shape()[self.feature_dim]
            pool_shape = self.input_shape[:self.feature_dim] + (out_feature_dim_size, self.pool_size) + self.input_shape[self.feature_dim + 1:]
            
            input_reshaped = input.reshape(pool_shape)
            output = T.max(input_reshaped, axis=self.feature_dim + 1)
        else:
            raise "Uknown implementation string '%s'" % self.implementation

        return output





def dump_params(l, **kwargs):
    """
    dump parameters from layer l and down into a file.
    The dump file has the same name as the script, with _paramdump.pkl appended.

    This dump can be used to recover after a crash.

    additional metadata (i.e. chunk number) can be passed as keyword arguments.
    """
    param_values = get_param_values(l)
    kwargs['param_values'] = param_values
    fn = os.path.basename(sys.argv[0]).replace(".py", "_paramdump.pkl")
    dir = os.path.dirname(sys.argv[0])
    path = os.path.join(dir, fn)

    with open(path, 'w') as f:
        pickle.dump(kwargs, f, pickle.HIGHEST_PROTOCOL)
