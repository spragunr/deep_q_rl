""" Utility to plot the first layer of convolutions learned by
the Deep q-network.

(Assumes dnn convolutions)

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
for f in range(w.shape[0]): # filters
    for c in range(w.shape[1]): # channels/time-steps
        plt.subplot(w.shape[0], w.shape[1], count)
        img = w[f, c, :, :]
        plt.imshow(img, vmin=img.min(), vmax=img.max(),
                   interpolation='none', cmap='gray')
        plt.xticks(())
        plt.yticks(())
        count += 1
plt.show()
