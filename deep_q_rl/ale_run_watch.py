""" This script runs a pre-trained network with the game
visualization turned on.

Usage:

ale_run_watch.py NETWORK_PKL_FILE
"""
import subprocess
import sys

command = ['./ale_run.py', '--glue-port', '4097', '--steps-per-epoch', '0',
           '--test-length', '10000', '--nn_file', sys.argv[1],
           '--pause', '.03', '--display-screen']

p1 = subprocess.Popen(command)

p1.wait()
