# Introduction 

This package provides a Theano-based implementation of the deep
Q-learning algorithm described in:

[Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

The neural network code is largely borrowed from [Sander Dieleman's
solution for the Galaxy Zoo Kaggle
challenge](http://benanne.github.io/2014/04/05/galaxy-zoo.html).

The results obtained with this code are not quite as good as the
results from the paper.  This could just be a matter of parameter
tuning (I've done very little) or it could be something more
fundamental.

Here is a video showing a trained network playing breakout:

 http://youtu.be/SZ88F82KLX4


# Dependencies

* A reasonably modern NVIDIA GPU
* Cython
* OpenCV
* Each of the following should be installed from the master branches on github:
  * [Theano](http://deeplearning.net/software/theano/) ([https://github.com/Theano/Theano](https://github.com/Theano/Theano))
  * [Pylearn2](http://deeplearning.net/software/pylearn2/) ([https://github.com/lisa-lab/pylearn2](https://github.com/lisa-lab/pylearn2))
  * [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) ([https://github.com/mgbellemare/Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment))

     We need a slightly modified version of ALE.  You'll need to replace `rlglue_controller.cpp` with the provided version before compiling.  This version handles down-sampling the image and converting to gray-scale.  It also updates the RLGlue interface to respect the "-restricted_action_set" flag. 
* [RL-Glue](http://glue.rl-community.org/wiki/Main_Page)
* [RL-Glue Python Codec](http://glue.rl-community.org/wiki/Python_Codec)


# Running

Use the script `ale_run.py` to start all the necessary processes:

`$ python ale_run.py --exp_pref data`

This will store output files in a folder prefixed with `data` in the current
directory.  Pickled version of the network objects are stored after every 
epoch.  The file `results.csv` will contain the testing output.  You can 
plot the progress by executing `plot_results.py`:

`$ python plot_results.py data_09-29-15-46_0p0001_0p9/results.csv`

After a couple of days, you can watch the trained network play using the 
`ale_run_watch.py` script: 

`$ python ale_run_watch.py data_09-29-15-46_0p0001_0p9/network_file_99.pkl`

# Getting Help

The [deep Q-learning web-forum](https://groups.google.com/forum/#!forum/deep-q-learning)
can be used for discussion and advice related to deep Q-learning in
general and this package in particular.

# See Also

* https://github.com/kristjankorjus/Replicating-DeepMind

  This is a work in progress.  Their project is described here: 
  http://robohub.org/artificial-general-intelligence-that-plays-atari-video-games-how-did-deepmind-do-it/

* https://github.com/muupan/dqn-in-the-caffe

  Working Caffe-based implementation.  (I haven't tried it, but there is a video of the agent playing Pong successfully.) 

* https://github.com/brian473/neural_rl

  This is an almost-working implementation developed last spring by my
  student Brian Brown.  I haven't reused his code, but Brian and I
  worked together to puzzle through some of the blank areas of the
  original paper.