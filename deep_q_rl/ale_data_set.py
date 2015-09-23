"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.

It allocates more memory than necessary, then shifts all of the
data back to 0 when the samples reach the end of the allocated memory.
"""

import numpy as np
import time
import theano

floatX = theano.config.floatX

class DataSet(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """
    def __init__(self, width, height, rng, max_steps=1000, phi_length=4):
        """Construct a NewDataSet.

        Arguments:
            width, height -- image size
            max_steps -- the number of time steps to store
            phi_length -- number of images to concatenate into a state
            rng -- initialized numpy random number generator, used to
            choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not
        # number of saved time steps.

        # Allocate the circular buffers and indices.
        imgs = np.zeros((max_steps, height, width), dtype='uint8')
        actions = np.zeros(max_steps, dtype='int32')
        rewards = np.zeros(max_steps, dtype=floatX)
        terminal = np.zeros(max_steps, dtype='bool')

        bottom = 0
        top = 0
        size = 0

        # Store arguments and circular buffers as member variables.
        self.__dict__.update(locals())
        del self.self

    def add_sample(self, img, action, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.imgs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img."""
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.imgs.take(indexes, axis=0, mode='wrap')
        phi[-1] = img
        return phi

    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.

        """
        count = 0

        # Allocate the response.
        states = np.zeros((batch_size, self.phi_length, self.height, self.width),
                          dtype='uint8')
        actions = np.zeros((batch_size, 1), dtype='int32')
        rewards = np.zeros((batch_size, 1), dtype=floatX)
        terminal = np.zeros((batch_size, 1), dtype='bool')
        next_states = np.zeros((batch_size, self.phi_length, self.height, self.width),
                               dtype='uint8')

        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            initial_indices = np.arange(index, index + self.phi_length)
            transition_indices = initial_indices + 1
            end_index = index + self.phi_length - 1
            
            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate.
            if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
                continue

            # Add the state transition.
            states[count] = self.imgs.take(initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            next_states[count] = self.imgs.take(transition_indices, axis=0, mode='wrap')
            count += 1

        return states, actions, rewards, next_states, terminal


# TESTING CODE BELOW THIS POINT...

def simple_tests():
    np.random.seed(222)
    dataset = DataSet(width=2, height=3, max_steps=6, phi_length=4, capacity=7)
    for i in range(10):
        img = np.random.randint(0, 256, size=(3, 2))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        print 'img', img
        dataset.add_sample(img, action, reward, terminal)
        print "S", dataset.states
        print "A", dataset.actions
        print "R", dataset.rewards
        print "T", dataset.terminal
        print "COUNT", "CAPACITY", dataset.count, dataset.capacity
        print
    print "LAST PHI", dataset.last_phi()
    print
    print 'BATCH', dataset.random_batch(2)


def speed_tests():

    dataset = DataSet(width=80, height=80, max_steps=20000, phi_length=4)

    img = np.random.randint(0, 256, size=(80, 80))
    action = np.random.randint(16)
    reward = np.random.random()
    start = time.time()
    for i in range(100000):
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset.add_sample(img, action, reward, terminal)
    print "samples per second: ", 100000 / (time.time() - start)

    start = time.time()
    for i in range(200):
        a = dataset.random_batch(32)
    print "batches per second: ", 200 / (time.time() - start)

    print dataset.last_phi()


def trivial_tests():

    dataset = DataSet(width=2, height=1, max_steps=3, phi_length=2)

    img1 = np.array([[1, 1]], dtype='uint8')
    img2 = np.array([[2, 2]], dtype='uint8')
    img3 = np.array([[3, 3]], dtype='uint8')

    dataset.add_sample(img1, 1, 1, False)
    dataset.add_sample(img2, 2, 2, False)
    dataset.add_sample(img3, 2, 2, True)
    print "last", dataset.last_phi()
    print "random", dataset.random_batch(1)


def max_size_tests():
    dataset1 = DataSet(width=3, height=4, max_steps=10, phi_length=4)
    dataset2 = DataSet(width=3, height=4, max_steps=1000, phi_length=4)
    for i in range(100):
        img = np.random.randint(0, 256, size=(4, 3))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset1.add_sample(img, action, reward, terminal)
        dataset2.add_sample(img, action, reward, terminal)
        np.testing.assert_array_almost_equal(dataset1.last_phi(),
                                             dataset2.last_phi())
        print "passed"


def test_iterator():
    dataset = DataSet(width=2, height=1, max_steps=10, phi_length=2)

    img1 = np.array([[1, 1]], dtype='uint8')
    img2 = np.array([[2, 2]], dtype='uint8')
    img3 = np.array([[3, 3]], dtype='uint8')
    img4 = np.array([[3, 3]], dtype='uint8')

    dataset.add_sample(img1, 1, 1, False)
    dataset.add_sample(img2, 2, 2, False)
    dataset.add_sample(img3, 3, 3, False)
    dataset.add_sample(img4, 4, 4, True)

    for s, a, r, t, ns in dataset.batch_iterator(2):
        print "s ", s, "a ",a, "r ",r,"t ", t,"ns ", ns


def test_random_batch():
    dataset1 = DataSet(width=3, height=4, max_steps=50, phi_length=4)
    dataset2 = DataSet(width=3, height=4, max_steps=50, phi_length=4,
                       capacity=2000)
    np.random.seed(hash(time.time()))

    for i in range(100):
        img = np.random.randint(0, 256, size=(4, 3))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True

        dataset1.add_sample(img, action, reward, terminal)
        dataset2.add_sample(img, action, reward, terminal)
        if i > 10:
            np.random.seed(i*11 * i)
            states1, actions1, rewards1, next_states1, terminals1 = \
                dataset1.random_batch(10)

            np.random.seed(i*11 * i)
            states2, actions2, rewards2, next_states2, terminals2 = \
                dataset2.random_batch(10)
            np.testing.assert_array_almost_equal(states1, states2)
            np.testing.assert_array_almost_equal(actions1, actions2)
            np.testing.assert_array_almost_equal(rewards1, rewards2)
            np.testing.assert_array_almost_equal(next_states1, next_states2)
            np.testing.assert_array_almost_equal(terminals1, terminals2)
            # if not np.array_equal(states1, states2):
            #     print states1,"\n", states2
            # if not np.array_equal(actions1, actions2):
            #     print actions1, "\n",actions2
            # if not np.array_equal(rewards1, rewards2):
            #     print rewards1, "\n",rewards2
            # if not np.array_equal(next_states1, next_states2):
            #     print next_states1, "\n",next_states2
            # if not np.array_equal(terminals1, terminals2):
            #     print terminals1, "\n",terminals2

            np.random.seed(hash(time.time()))


def test_memory_usage_ok():
    import memory_profiler
    dataset = DataSet(width=80, height=80, max_steps=100000, phi_length=4)
    last = time.time()

    for i in xrange(1000000000):
        if (i % 100000) == 0:
            print i
        dataset.add_sample(np.random.random((80, 80)), 1, 1, False)
        if i > 200000:
            states, actions, rewards, next_states, terminals = \
                                        dataset.random_batch(32)
        if (i % 10007) == 0:
            print time.time() - last
            mem_usage = memory_profiler.memory_usage(-1)
            print len(dataset), mem_usage
        last = time.time()


def main():
    #speed_tests()
    test_memory_usage_ok()
    #test_random_batch()
    #max_size_tests()
    #simple_tests()
    #test_iterator()

if __name__ == "__main__":
    main()
