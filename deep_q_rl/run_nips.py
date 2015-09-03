#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

"""
from ale_agent import NeuralAgent
from ale_parameters_default import ParametersDefault
from q_network import DeepQLearner

import launcher
import sys


class Parameters:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    steps_per_epoch = 50000
    epochs = 100
    steps_per_test = 10000

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
    update_rule = 'rmsprop'
    batch_accumulator = 'mean'
    learning_rate = .0002
    discount = .95
    rms_decay = .99 # (Rho)
    rms_epsilon = 1e-6
    momentum = 0
    clip_delta = 0
    epsilon_start = 1.0
    epsilon_min = .1
    epsilon_decay = 1000000
    phi_length = 4
    update_frequency = 1
    replay_memory_size = 1000000
    batch_size = 32
    network_type = "nips_dnn"
    freeze_interval = -1
    input_scale = 255.
    replay_start_size = 100
    resize_method = 'crop'
    resized_width = 84
    resized_height = 84
    death_ends_episode = 'false'
    max_start_nullops = 0
    deterministic = 'true'
    cudnn_deterministic = 'false'

    agent_type = NeuralAgent
    qlearner_type = DeepQLearner

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Parameters, __doc__)
