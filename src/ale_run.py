"""This script launches all of the processes necessary to train a
deep Q-network on an ALE game.

Usage:

ale_run.py [--glue_port GLUE_PORT]

All unrecognized command line arguments will be passed on to
rl_glue_ale_agent.py
"""
import subprocess
import sys
import os
import argparse

ROM_PATH = "/home/spragunr/neural_rl_libraries/roms/breakout.bin"

# Build shift.so if necessary...
if not os.path.isfile('shift.so'):
    subprocess.Popen('python setup.py build_ext --inplace', shell=True)

# Check for glue_port command line argument and set it up...
parser = argparse.ArgumentParser(description='Neural rl agent.')
parser.add_argument('--glue_port', type=str, default="4096",
                    help='rlglue port (default 4096)')
args, unknown = parser.parse_known_args()
my_env = os.environ.copy()
my_env["RLGLUE_PORT"] = args.glue_port


# Start the necessary processes:
p1 = subprocess.Popen(['rl_glue'], env=my_env)
p2 = subprocess.Popen('ale -game_controller rlglue -frame_skip 4 '+ ROM_PATH,
                      shell=True, env=my_env)
p3 = subprocess.Popen(['./rl_glue_ale_experiment.py'], env=my_env)
p4 = subprocess.Popen(['./rl_glue_ale_agent.py'] + sys.argv[1:], env=my_env)

p1.wait()
p2.wait()
p3.wait()
p4.wait()
