# Introduction 

This package provides a Lasagne/Theano-based implementation of the deep
Q-learning algorithm described in:

[Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

and 

Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

Here is a video showing a trained network playing breakout (using an earlier version of the code):

 http://youtu.be/SZ88F82KLX4

# Dependencies

* A reasonably modern NVIDIA GPU
* OpenCV
* [Theano](http://deeplearning.net/software/theano/) ([https://github.com/Theano/Theano](https://github.com/Theano/Theano))
* [Lasagne](http://lasagne.readthedocs.org/en/latest/) ([https://github.com/Lasagne/Lasagne](https://github.com/Lasagne/Lasagne)
* [Pylearn2](http://deeplearning.net/software/pylearn2/) ([https://github.com/lisa-lab/pylearn2](https://github.com/lisa-lab/pylearn2))
* [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) ([https://github.com/mgbellemare/Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment))

The script `dep_script.sh` can be used to install all dependencies under Ubuntu.


# Running

Use the scripts `run_nips.py` or `run_nature.py` to start all the necessary processes:

`$ ./run_nips.py --rom breakout`

`$ ./run_nature.py --rom breakout`

The `run_nips.py` script uses parameters consistent with the original
NIPS workshop paper.  This code should take 2-4 days to complete.  The
`run_nature.py` script uses parameters consistent with the Nature
paper.  The final policies should be better, but it will take 6-10
days to finish training.

Either script will store output files in a folder prefixed with the
name of the ROM.  Pickled version of the network objects are stored
after every epoch.  The file `results.csv` will contain the testing
output.  You can plot the progress by executing `plot_results.py`:

`$ python plot_results.py breakout_05-28-17-09_0p00025_0p99/results.csv`

After training completes, you can watch the network play using the 
`ale_run_watch.py` script: 

`$ python ale_run_watch.py breakout_05-28-17-09_0p00025_0p99/network_file_99.pkl`

# Performance Tuning

## Theano Configuration

Setting `allow_gc=False` in `THEANO_FLAGS` or in the `.theanorc` file
significantly improves performance at the expense of a slight increase
in memory usage on the GPU.


# Getting Help

The [deep Q-learning web-forum](https://groups.google.com/forum/#!forum/deep-q-learning)
can be used for discussion and advice related to deep Q-learning in
general and this package in particular.

# See Also

* https://sites.google.com/a/deepmind.com/dqn

  This is the code DeepMind used for the Nature paper.  The license
  only permits the code to be used for "evaluating and reviewing" the
  claims made in the paper.

* https://github.com/muupan/dqn-in-the-caffe

  Working Caffe-based implementation.  (I haven't tried it, but there
  is a video of the agent playing Pong successfully.)

* https://github.com/kristjankorjus/Replicating-DeepMind

  Defunct?  As far as I know, this package was never fully functional.  The project is described here: 
  http://robohub.org/artificial-general-intelligence-that-plays-atari-video-games-how-did-deepmind-do-it/

* https://github.com/brian473/neural_rl

  This is an almost-working implementation developed during Spring
  2014 by my student Brian Brown.  I haven't reused his code, but
  Brian and I worked together to puzzle through some of the blank
  areas of the original paper.

