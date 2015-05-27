"""
This uses the skeleton_agent.py file from the Python-codec of rl-glue
as a starting point.


Author: Nathan Sprague
"""

#
# Copyright (C) 2008, Brian Tanner
#
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import copy
import os
import cPickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
import time
import logging

import random
import numpy as np
import cv2

import argparse

import matplotlib.pyplot as plt

import ale_data_set
import theano
from network import DeepQLearner

import sys
sys.setrecursionlimit(10000)

floatX = theano.config.floatX

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 210

CROPPED_WIDTH = 84
CROPPED_HEIGHT = 84

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games. 
CROP_OFFSET = 8


class NeuralAgent(Agent):
    randGenerator=random.Random()

    def __init__(self, discount, learning_rate, rms_decay, momentum,
                 epsilon_start, epsilon_min, epsilon_decay,
                 phi_length, replay_memory_size, exp_pref, nn_file,
                 pause, network_type, freeze_interval, batch_size,
                 replay_start_size, update_frequency):

        self.discount = discount
        self.learning_rate = learning_rate
        self.rms_decay = rms_decay
        self.momentum = momentum
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.phi_length = phi_length
        self.replay_memory_size = replay_memory_size
        self.exp_pref = exp_pref
        self.nn_file = nn_file
        self.pause = pause
        self.network_type = network_type
        self.freeze_interval = freeze_interval
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.image_resize = 'scale'

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                        "{}".format(self.learning_rate).replace(".", "p") + \
                        "_" + "{}".format(self.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except:
            os.makedirs(self.exp_dir)



    def agent_init(self, task_spec_string):
        """
        This function is called once at the beginning of an experiment.

        Arguments: task_spec_string - A string defining the task.  This string
                                      is decoded using
                                      TaskSpecVRLGLUE3.TaskSpecParser
        """
        # DO SOME SANITY CHECKING ON THE TASKSPEC
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_string)
        if TaskSpec.valid:

            assert ((len(TaskSpec.getIntObservations()) == 0) !=
                    (len(TaskSpec.getDoubleObservations()) == 0)), \
                "expecting continous or discrete observations.  Not both."
            assert len(TaskSpec.getDoubleActions()) == 0, \
                "expecting no continuous actions"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), \
                " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), \
                " expecting max action to be a number not a special value"
            self.num_actions = TaskSpec.getIntActions()[0][1]+1
        else:
            logging.error("INVALID TASK SPEC")

        self.data_set = ale_data_set.DataSet(width=CROPPED_WIDTH,
                                             height=CROPPED_HEIGHT,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=CROPPED_WIDTH,
                                                  height=CROPPED_HEIGHT,
                                                  max_steps=10,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0
            
        #self.target_reset_freq = 10000 # target network update frequency
        self.testing = False

        if self.nn_file is None:
            self.network = self._init_network()
        else:
            handle = open(self.nn_file, 'r')
            self.network = cPickle.load(handle)

        self._open_results_file()
        self._open_learning_file()

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None



    def _init_network(self):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        return DeepQLearner(CROPPED_WIDTH, 
                            CROPPED_HEIGHT, 
                            self.num_actions, 
                            self.phi_length, 
                            self.discount,
                            self.learning_rate,
                            self.rms_decay,
                            self.momentum,
                            self.freeze_interval,
                            self.batch_size,
                            self.network_type)



    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q\n')

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                  self.total_reward / float(num_episodes),
                                  holdout_sum)
        self.results_file.write(out)


    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)


    def agent_start(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - An observation of type rlglue.types.Observation

        Returns:
           An action of type rlglue.types.Action
        """

        self.step_counter = 0
        self.batch_counter = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        this_int_action = self.randGenerator.randint(0, self.num_actions-1)
        return_action = Action()
        return_action.intArray = [this_int_action]

        self.last_action = copy.deepcopy(return_action)

        self.last_img = self._resize_observation(observation.intArray)

        return return_action


    def _show_phis(self, phi1, phi2):
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def _resize_observation(self, observation):
        # reshape linear to original image size, skipping the RAM portion
        image = observation[128:].reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        # convert from int32s
        image = np.array(image, dtype="uint8")

        # convert to greyscale
        greyscaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.image_resize == 'crop':
            # resize keeping aspect ratio
            resize_width = CROPPED_WIDTH
            resize_height = int(round(float(IMAGE_HEIGHT) * CROPPED_HEIGHT /
                                      IMAGE_WIDTH))

            resized = cv2.resize(greyscaled, (resize_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - CROPPED_HEIGHT
            cropped = resized[crop_y_cutoff:crop_y_cutoff + CROPPED_HEIGHT, :]

            return cropped
        elif self.image_resize == 'scale':
            return cv2.resize(greyscaled, (CROPPED_WIDTH, CROPPED_HEIGHT),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')


    def agent_step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - An observation of type rlglue.types.Observation

        Returns:
           An action of type rlglue.types.Action

        """

        self.step_counter += 1
        return_action = Action()

        cur_img = self._resize_observation(observation.intArray)

        #TESTING---------------------------
        if self.testing:
            self.total_reward += reward
            int_action = self._choose_action(self.test_data_set, .05,
                                             cur_img, np.clip(reward, -1, 1))
            if self.pause > 0:
                time.sleep(self.pause)

        #NOT TESTING---------------------------
        else:

            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                int_action = self._choose_action(self.data_set, self.epsilon,
                                                 cur_img,
                                                 np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    self.batch_counter += 1
                    self.loss_averages.append(loss)

            else: # Still gathering initial random data...
                int_action = self._choose_action(self.data_set, 1.0,
                                                 cur_img,
                                                 np.clip(reward, -1, 1))

        return_action.intArray = [int_action]

        self.last_action = copy.deepcopy(return_action)
        self.last_img = cur_img

        return return_action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img,
                            self.last_action.intArray[0],
                            reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            int_action = self.network.choose_action(phi, epsilon)
        else:
            int_action = self.randGenerator.randint(0, self.num_actions - 1)

        return int_action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, actions, rewards, next_states, terminals = \
                                self.data_set.random_batch(self.batch_size)
        return self.network.train(states, actions, rewards,
                                  next_states, terminals)


    def agent_end(self, reward):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.

        Returns:
            None
        """
        self.episode_counter += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            self.total_reward += reward
        else:

            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action.intArray[0],
                                     np.clip(reward, -1, 1),
                                     True)

            if self.batch_counter > 0:
                logging.info(
                    "Batches/second: {:.2f}  Average loss: {:.4f}".format(\
                            self.batch_counter/total_time,
                            np.mean(self.loss_averages)))

                self._update_learning_file()



    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """
        pass

    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file.
        """

        #WE NEED TO DO THIS BECAUSE agent_end is not called
        # we run out of steps.
        if in_message.startswith("episode_end"):
            self.agent_end(0)

        elif in_message.startswith("finish_epoch"):
            epoch = int(in_message.split(" ")[1])
            net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                            '.pkl', 'w')
            cPickle.dump(self.network, net_file, -1)
            net_file.close()

        elif in_message.startswith("start_testing"):
            self.testing = True
            self.total_reward = 0
            self.episode_counter = 0

        elif in_message.startswith("finish_testing"):
            self.testing = False
            holdout_size = 3200
            epoch = int(in_message.split(" ")[1])

            if self.holdout_data is None:
                self.holdout_data = self.data_set.random_batch(holdout_size)[0]

            holdout_sum = 0
            for i in range(holdout_size):
                holdout_sum += np.mean(
                    self.network.q_vals(self.holdout_data[i, ...]))

            self._update_results_file(epoch, self.episode_counter,
                                      holdout_sum / holdout_size)
        else:
            return "I don't know how to respond to your message"

def main():
    AgentLoader.loadAgent(NeuralAgent())


if __name__ == "__main__":
    main()
