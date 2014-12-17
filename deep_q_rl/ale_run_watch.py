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

ROM_PATH = "/home/spragunr/neural_rl_libraries/roms/breakout.bin"

p1 = subprocess.Popen(['rl_glue'], env=my_env)
p2 = subprocess.Popen('ale -display_screen true -game_controller rlglue -frame_skip 4 '+ ROM_PATH, shell=True, env=my_env)
#p3 = subprocess.Popen(['./rl_glue_watch_experiment.py'], env=my_env)
p3 = subprocess.Popen(['./rl_glue_ale_experiment.py', '--epoch_length', '0', 
                       '--test_length', '10000'], env=my_env)
p4 = subprocess.Popen(['./rl_glue_ale_agent.py', "--pause", ".03", 
                       "--nn_file", sys.argv[1]],
                      env=my_env)

p1.wait()
p2.wait()
p3.wait()
p4.wait()
