#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015

"""
import sys
from ale_agent import NeuralAgent

from ale_parameters_default import ParametersDefault
from q_network import DeepQLearner
import launcher


class Parameters:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    steps_per_epoch = 250000
    epochs = 200
    steps_per_test = 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    base_rom_path = "../roms/"
    rom = 'breakout.bin'
    frame_skip = 4
    repeat_action_probability = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    update_rule = 'deepmind_rmsprop'
    batch_accumulator = 'sum'
    learning_rate = .00025
    discount = .99
    rms_decay = .95 # (Rho)
    rms_epsilon = .01
    momentum = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    clip_delta = 1.0
    epsilon_start = 1.0
    epsilon_min = .1
    epsilon_decay = 1000000
    phi_length = 4
    update_frequency = 4
    replay_memory_size = 1000000
    batch_size = 32
    network_type = "nature_dnn"
    freeze_interval = 10000
    input_scale = 255.
    replay_start_size = 50000
    resize_method = 'scale'
    resized_width = 84
    resized_height = 84
    death_ends_episode = 'true'
    max_start_nullops = 30
    deterministic = 'true'
    cudnn_deterministic = 'false'

    agent_type = NeuralAgent
    qlearner_type = DeepQLearner

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Parameters, __doc__)
