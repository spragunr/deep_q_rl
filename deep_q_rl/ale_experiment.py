"""
This is an RLGlue experiment designed to collect the type of data
presented in:

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

(Based on the sample_experiment.py from the Rl-glue python codec examples.)


Author: Nathan Sprague

"""
import logging
import numpy as np
import cv2

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8


class ALEExperiment(object):
    def __init__(self, ale, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length):
        self.ale = ale
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.min_action_set = ale.getMinimalActionSet()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.width, self.height = ale.getScreenDims()
        self.screenRGB = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length, "training")
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing()
                self.run_epoch(epoch, self.test_length, "testing")
                self.agent.finish_testing(epoch)

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
            _, num_steps = self.run_episode(steps_left)

            steps_left -= num_steps

    def run_episode(self, max_steps):
        """ return (terminal, num_steps)"""
        self.ale.reset_game()
        num_steps = 1

        action = self.agent.start_episode(self.get_image())

        while not self.ale.game_over() and num_steps < max_steps:
            reward = self.ale.act(self.min_action_set[action])
            action = self.agent.step(reward, self.get_image())
            num_steps += 1

        self.agent.end_episode(reward)
        return self.ale.game_over(), num_steps


    def get_image(self):
        """ Get a screen image from ale and resize appropriately. """

        # convert to greyscale
        self.ale.getScreenRGB(self.screenRGB)

        greyscaled = cv2.cvtColor(self.screenRGB, cv2.COLOR_RGB2GRAY)

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(greyscaled,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
                              crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'scale':
            return cv2.resize(greyscaled,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')
