#! /usr/bin/env python
"""
Execute a training run of using an agent that plays random moves

"""
import sys
from ale_agent_random import AgentRandom

from ale_parameters_default import ParametersDefault
import launcher


class Parameters(ParametersDefault):
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000
    EPOCHS = 200
    STEPS_PER_TEST = 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../roms/"
    ROM = 'breakout.bin'
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95  # (Rho)
    RMS_EPSILON = .01
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nature_dnn"
    FREEZE_INTERVAL = 10000
    INPUT_SCALE = 255.
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30

    AGENT_TYPE = AgentRandom
    QLEARNER_TYPE = None

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Parameters, __doc__)
