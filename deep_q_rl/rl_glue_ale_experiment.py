"""
This is an RLGlue experiment designed to collect the type of data
presented in:

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

(Based on the sample_experiment.py from the Rl-glue python codec examples.)


Author: Nathan Sprague

"""
import rlglue.RLGlue as RLGlue
import logging

class AleExperiment(object):
    def __init__(self, num_epochs, epoch_length, test_length):
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        RLGlue.RL_init()

        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length, "training")
            RLGlue.RL_agent_message("finish_epoch " + str(epoch))

            if self.test_length > 0:
                RLGlue.RL_agent_message("start_testing")
                self.run_epoch(epoch, self.test_length, "testing")
                RLGlue.RL_agent_message("finish_testing " + str(epoch))
        
    def run_epoch(self, epoch, num_steps, prefix):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        num_steps - steps per epoch
        prefix - string to print ('training' or 'testing')

        """
        steps_left = num_steps
        while steps_left > 0:
            logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(steps_left))
            terminal = RLGlue.RL_episode(steps_left)
            if not terminal:
                RLGlue.RL_agent_message("episode_end")
            steps_left -= RLGlue.RL_num_steps()

