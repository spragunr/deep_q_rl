#! /usr/bin/env python

import launcher
import sys

class Defaults:
    # ----------------------
    # RLGlue Parameters
    # ----------------------
    RLGLUE_PORT = 4096
    
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 50000
    EPOCHS = 200
    STEPS_PER_TEST = 10000
    
    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../roms/"
    ROM = 'breakout.bin'
    FRAME_SKIP = 4
    
    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    MOMENTUM = .95
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4 # Not yet used.
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE="nature_dnn"
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000 # Not yet used


if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults)
