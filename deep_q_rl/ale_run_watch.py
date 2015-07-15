""" This script runs a pre-trained network with the game
visualization turned on.

Usage:

ale_run_watch.py NETWORK_PKL_FILE [ ROM ]
"""
import subprocess
import sys

def run_watch():
    command = ['./run_nature.py', '--steps-per-epoch', '0',
               '--test-length', '10000', '--nn-file', sys.argv[1],
               '--display-screen']

    if len(sys.argv) > 2:
        command.extend(['--rom', sys.argv[2]])

    p1 = subprocess.Popen(command)
    
    p1.wait()

if __name__ == "__main__":
    run_watch()
