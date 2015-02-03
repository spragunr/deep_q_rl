""" This script runs a pre-trained network with the game
visualization turned on.

Usage:

ale_run_watch.py NETWORK_PKL_FILE
"""
import subprocess
import os
import sys

my_env = os.environ.copy()
my_env["RLGLUE_PORT"] = "4097"

# Put your binaries under the directory 'deep_q_rl/roms'
ROM_PATH = "../roms/breakout.bin"

p1 = subprocess.Popen(['rl_glue'], env=my_env)
ale_string = ("ale -game_controller rlglue -frame_skip 4  "
              "-restricted_action_set true -display_screen true "
              "-disable_color_averaging true " + ROM_PATH)
p2 = subprocess.Popen(ale_string, shell=True, env=my_env)
p3 = subprocess.Popen(['./rl_glue_ale_experiment.py', '--epoch_length', '0', 
                       '--test_length', '10000'], env=my_env)
p4 = subprocess.Popen(['./rl_glue_ale_agent.py', "--pause", ".03", 
                       "--nn_file", sys.argv[1]],
                      env=my_env)

p1.wait()
p2.wait()
p3.wait()
p4.wait()
