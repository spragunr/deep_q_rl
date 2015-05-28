""" Utility to plot the first layer of convolutions learned by
the Deep q-network.

(Assumes cuda_convnet layers)

Usage:

plot_filters.py PICKLED_NN_FILE
"""

import sys
import matplotlib.pyplot as plt
import cPickle
import lasagne.layers

net_file = open(sys.argv[1], 'r')
network = cPickle.load(net_file)
print network
q_layers = lasagne.layers.get_all_layers(network.l_out)
w = q_layers[1].W.get_value()
count = 1
for f in range(w.shape[3]):
    for c in range(w.shape[0]):
        plt.subplot(w.shape[3], w.shape[0], count)
        img = w[c, :, :, f]
        plt.imshow(img, vmin=img.min(), vmax=img.max(),
                   interpolation='none', cmap='gray')
        plt.xticks(())
        plt.yticks(())
        count += 1
plt.show()
