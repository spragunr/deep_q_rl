"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.

It allocates more memory than necessary, then shifts all of the
data back to 0 when the samples reach the end of the allocated memory.
"""

import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
import shift
import time
import theano

floatX = theano.config.floatX

class DataSet(object):
    """ Class represents a data set that stores a fixed-length history.
    """

    def __init__(self, width, height, rng, max_steps=1000, phi_length=4,
                 capacity=None):
        """  Construct a DataSet.

        Arguments:
            width,height - image size
            max_steps - the length of history to store.
            phi_length - number of images to concatenate into a state.
            rng        - initialized numpy random number generator.
            capacity - amount of memory to allocate (just for debugging.)
        """

        self.rng = rng
        self.count = 0
        self.max_steps = max_steps
        self.phi_length = phi_length
        if capacity == None:
            self.capacity = max_steps + int(np.ceil(max_steps * .1))
        else:
            self.capacity = capacity
        self.states = np.zeros((self.capacity, height, width), dtype='uint8')
        self.actions = np.zeros(self.capacity, dtype='int32')
        self.rewards = np.zeros(self.capacity, dtype=floatX)
        self.terminal = np.zeros(self.capacity, dtype='bool')


    def _min_index(self):
        return max(0, self.count - self.max_steps)

    def _max_index(self):
        return self.count - (self.phi_length + 1)

    def __len__(self):
        """ Return the total number of avaible data items. """
        return max(0, (self._max_index() - self._min_index()) + 1)

    def add_sample(self, state, action, reward, terminal):
        self.states[self.count, ...] = state
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.terminal[self.count] = terminal
        self.count += 1

        # Shift the final max_steps back to the beginning.
        if self.count == self.capacity:
            roll_amount = self.capacity - self.max_steps
            shift.shift3d_uint8(self.states, roll_amount)
            self.actions = np.roll(self.actions, -roll_amount)
            self.rewards = np.roll(self.rewards, -roll_amount)
            self.terminal = np.roll(self.terminal, -roll_amount)
            self.count = self.max_steps

    def single_episode(self, start, end):
        """ Make sure that a possible phi does not cross a trial boundary.
        """
        return np.alltrue(np.logical_not(self.terminal[start:end]))

    def last_phi(self):
        """
        Return the most recent phi.
        """
        phi = self._make_phi(self.count - self.phi_length)
        return  np.array(phi, dtype=floatX)

    def phi(self, state):
        """
        Return a phi based on the latest image, by grabbing enough
        history from the data set to fill it out.
        """
        phi = np.empty((self.phi_length,
                        self.states.shape[1],
                        self.states.shape[2]),
                       dtype=floatX)

        phi[0:(self.phi_length-1), ...] = self.last_phi()[1::]
        phi[self.phi_length-1, ...] = state
        return phi

    def _make_phi(self, index):
        end_index = index + self.phi_length - 1
        #assert self.single_episode(index, end_index)
        return self.states[index:end_index + 1, ...]

    def _empty_batch(self, batch_size):
        # Set aside memory for the batch
        states = np.empty((batch_size, self.phi_length,
                           self.states.shape[1], self.states.shape[2]),
                          dtype=floatX)
        actions = np.empty((batch_size, 1), dtype='int32')
        rewards = np.empty((batch_size, 1), dtype=floatX)
        terminals = np.empty((batch_size, 1), dtype=bool)

        next_states = np.empty((batch_size, self.phi_length,
                                self.states.shape[1],
                                self.states.shape[2]), dtype=floatX)
        return states, actions, rewards, terminals, next_states

    def batch_iterator(self, batch_size):
        """ Generator for iterating over all valid batches. """
        index = self._min_index()
        batch_count = 0
        states, actions, rewards, terminals, next_states = \
                self._empty_batch(batch_size)      
        while index <= self._max_index():
            end_index = index + self.phi_length - 1
            if self.single_episode(index, end_index):
                states[batch_count, ...] = self._make_phi(index)
                actions[batch_count, 0] = self.actions[end_index]
                rewards[batch_count, 0] = self.rewards[end_index]
                terminals[batch_count, 0] = self.terminal[end_index]
                next_states[batch_count, ...] = self._make_phi(index + 1)
                batch_count += 1
            index += 1

            if batch_count == batch_size:
                yield states, actions, rewards, terminals, next_states
                batch_count = 0
                states, actions, rewards, terminals, next_states = \
                    self._empty_batch(batch_size)      

                
    def random_batch(self, batch_size):

        count = 0
        states, actions, rewards, terminals, next_states = \
            self._empty_batch(batch_size)

        # Grab random samples until we have enough
        while count < batch_size:
            index = self.rng.randint(self._min_index(), self._max_index()+1)
            end_index = index + self.phi_length - 1
            if self.single_episode(index, end_index):
                states[count, ...] = self._make_phi(index)
                actions[count, 0] = self.actions[end_index]
                rewards[count, 0] = self.rewards[end_index]
                terminals[count, 0] = self.terminal[end_index]
                next_states[count, ...] = self._make_phi(index + 1)
                count += 1

        return states, actions, rewards, next_states, terminals



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
