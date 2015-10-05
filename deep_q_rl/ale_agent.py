"""
The NeuralAgent class wraps a deep Q-network for training and testing
in the Arcade learning environment.

Author: Nathan Sprague

"""

import os
import cPickle
import datetime
import logging
import json

import numpy as np

from ale_agent_base import AgentBase
import ale_data_set

import sys

sys.setrecursionlimit(10000)


class NeuralAgent(AgentBase):
    def __init__(self, params):
        super(NeuralAgent, self).__init__(params)

        self.params = params
        self.network = None
        self.action_set = None
        self.num_actions = -1

        self.epsilon_start = self.params.epsilon_start
        self.epsilon_min = self.params.epsilon_min
        self.epsilon_decay = self.params.epsilon_decay
        self.replay_memory_size = self.params.replay_memory_size
        self.exp_pref = self.params.experiment_prefix
        self.replay_start_size = self.params.replay_start_size
        self.update_frequency = self.params.update_frequency
        self.phi_length = self.params.phi_length
        self.image_width = self.params.resized_width
        self.image_height = self.params.resized_height

        self.rng = self.params.rng

        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=self.rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=self.rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self.current_epoch = 0
        self.episode_counter = 0
        self.batch_counter = 0
        self.total_reward = 0
        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        self.export_dir = self._create_export_dir()
        self._open_params_file()
        self._open_results_file()
        self._open_learning_file()

    def initialize(self, action_set):
        self.action_set = action_set
        self.num_actions = len(self.action_set)

        if self.params.qlearner_type is None:
            raise Exception("The QLearner/network type has not been specified")

        if self.params.nn_file is None:
            self.network = self.params.qlearner_type(
                num_actions=self.num_actions,
                input_width=self.params.resized_width,
                input_height=self.params.resized_height,
                num_frames=self.params.phi_length,
                params=self.params)
        else:
            handle = open(self.params.nn_file, 'r')
            self.network = cPickle.load(handle)

    # region Dumping/Logging
    def _create_export_dir(self):
        # CREATE A FOLDER TO HOLD RESULTS
        # this is now just exp_pref + timestamp. params are in params.json
        time_str = datetime.datetime.now().strftime("_%m-%d-%H%M_%S_%f")
        export_dir = self.exp_pref + time_str
        try:
            os.stat(export_dir)
        except OSError:
            os.makedirs(export_dir)

        return export_dir

    def _open_params_file(self):
        self.params_file = open(self.export_dir + '/params.json', 'w')
        param_dict = {k:v for k, v in self.params.__dict__.items() \
                      if "__" not in k \
                      and isinstance(v, (int, float, str, bool))}
        json.dump(param_dict, self.params_file, indent=4)
        self.params_file.close()

    def _open_results_file(self):
        logging.info("OPENING " + self.export_dir + '/results.csv')
        self.results_file = open(self.export_dir + '/results.csv', 'w', 0)
        self.results_file.write(
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.export_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{}\n".format(epoch, num_episodes,
                                        self.total_reward,
                                        self.total_reward / float(num_episodes),
                                        holdout_sum)

        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def _persist_network(self, network_filename):
        full_filename = os.path.join(self.export_dir, network_filename)
        with open(full_filename, 'w') as net_file:
            cPickle.dump(self.network, net_file, -1)

    # endregion

    def start_epoch(self, epoch):
        self.current_epoch = epoch

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p + 1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p + 5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def _step_testing(self, reward, observation):
        action = self._choose_action(data_set=self.test_data_set,
                                     epsilon=.05,
                                     cur_img=observation,
                                     reward=np.clip(reward, -1, 1))
        return action

    def _step_training(self, reward, observation):
        if len(self.data_set) > self.replay_start_size:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon - self.epsilon_rate)

            action = self._choose_action(data_set=self.data_set,
                                         epsilon=self.epsilon,
                                         cur_img=observation,
                                         reward=np.clip(reward, -1, 1))

            if self.step_counter % self.update_frequency == 0:
                loss = self._do_training()
                self.batch_counter += 1
                self.loss_averages.append(loss)

        else:  # Still gathering initial random data...
            action = self._choose_action(data_set=self.data_set,
                                         epsilon=self.epsilon,
                                         cur_img=observation,
                                         reward=np.clip(reward, -1, 1))
        return action

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """
        self.episode_reward += reward
        if self.testing:
            action = self._step_testing(reward, observation)
        else:
            action = self._step_training(reward, observation)

        self.step_counter += 1
        self.last_action = action
        self.last_img = observation

        return action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, actions, rewards, next_states, terminals = \
            self.data_set.random_batch(
                self.network.batch_size)
        return self.network.train(states, actions, rewards,
                                  next_states, terminals)

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:

            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)

            if self.batch_counter > 0:
                self._update_learning_file()
                logging.info(
                    "average loss: {:.4f}".format(np.mean(self.loss_averages)))

    def finish_epoch(self, epoch):
        network_filename = 'network_file_' + str(epoch) + '.pkl'
        self._persist_network(network_filename)

    def start_testing(self, epoch):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            self.holdout_data = self.data_set.random_batch(holdout_size)[0]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i, ...]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)


if __name__ == "__main__":
    pass
