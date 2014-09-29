#!/usr/bin/env python
"""
This is an RLGlue experiment designed to collect the type of data
presented in:

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

(Based on the sample_experiment.py from the Rl-glue python codec examples.)

usage: rl_glue_ale_experiment.py [-h] [--num_epochs NUM_EPOCHS]
                                 [--epoch_length EPOCH_LENGTH]
                                 [--test_length TEST_LENGTH]

Author: Nathan Sprague

"""
import rlglue.RLGlue as RLGlue
import argparse

def run_epoch(epoch, num_steps, prefix):
    """ Run one 'epoch' of training or testing, where an epoch is defined
    by the number of steps executed.  Prints a progress report after
    every trial

    Arguments:
       num_steps - steps per epoch
       prefix - string to print ('training' or 'testing')

    """
    steps_left = num_steps
    while steps_left > 0:
        print prefix + " epoch: ", epoch, "steps_left: ", steps_left
        terminal = RLGlue.RL_episode(steps_left)
        if not terminal:
            RLGlue.RL_agent_message("episode_end")
        steps_left -= RLGlue.RL_num_steps()


def main():
    """
    Run the desired number of training epochs, a testing epoch
    is conducted after each training epoch.
    """

    parser = argparse.ArgumentParser(description='Neural rl experiment.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--epoch_length', type=int, default=50000,
                        help='Number of steps per epoch')
    parser.add_argument('--test_length', type=int, default=10000,
                        help='Number of steps per test')
    args = parser.parse_args()

    RLGlue.RL_init()

    for epoch in range(1, args.num_epochs + 1):
        RLGlue.RL_agent_message("training")
        run_epoch(epoch, args.epoch_length, "training")

        RLGlue.RL_agent_message("start_testing")
        run_epoch(epoch, args.test_length, "testing")
        RLGlue.RL_agent_message("finish_testing " + str(epoch))


if __name__ == "__main__":
    main()
