""" Utility to plot the first layer of convolutions learned by
the Deep q-network.

Usage:

plot_filters.py PICKLED_NN_FILE
"""

import sys
import matplotlib.pyplot as plt
import cPickle

net_file = open(sys.argv[1], 'r')
network = cPickle.load(net_file)
print network
w = network.q_layers[2].W.get_value()
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
