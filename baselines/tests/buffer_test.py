import unittest
from baselines import Buffer

import numpy as np

class BufferTest(unittest.TestCase):
    def test_can_sample(self):
        # env = gym.make('MountainCarContinuous-v0')

        size = 2
        action_space_size = 3
        observation_space_size = 1

        b = Buffer(size, action_space_size, observation_space_size)

        self.assertFalse(b.can_sample(1))
        self.assertFalse(b.can_sample(2))

        b.add(np.ones(observation_space_size), np.ones(action_space_size), 0,np.ones(observation_space_size), False)

        self.assertTrue(b.can_sample(1))
        self.assertFalse(b.can_sample(2))

        b.add(np.ones(observation_space_size), np.ones(action_space_size), 0, np.ones(observation_space_size), False)
        self.assertTrue(b.can_sample(1))
        self.assertTrue(b.can_sample(2))

        b.add(np.ones(observation_space_size), np.ones(action_space_size), 0, np.ones(observation_space_size), False)
        self.assertFalse(b.can_sample(3))

    def test_sample(self):
        size = 2
        action_space_size = 3
        observation_space_size = 1

        b = Buffer(size, action_space_size, observation_space_size)

        b.add(np.ones(observation_space_size), np.ones(action_space_size), 0, np.ones(observation_space_size), False)
        b.add(np.ones(observation_space_size), np.ones(action_space_size), 0, np.ones(observation_space_size), False)

        s0,a,r,s1,d = b.sample(2)

        self.assertEqual(s0.shape, (2,observation_space_size))
        self.assertEqual(s1.shape, (2,observation_space_size))
        self.assertEqual(a.shape, (2,action_space_size))
        self.assertEqual(r.shape, (2,1))
        self.assertEqual(d.shape, (2,1))

        self.assertEqual(d.dtype,np.float32)

    def test_sample_stochasticity(self):
        size = 5
        action_space_size = 3
        observation_space_size = 1

        b = Buffer(size, action_space_size, observation_space_size)

        for i in range(size):
            b.add(np.ones(observation_space_size)*i, np.ones(action_space_size)*i, i, np.ones(observation_space_size)*i, False)

        counters = np.zeros((size,1))

        sample_size = 2
        sample_count = 10000

        for i in range(sample_count):
            s0, a, r, s1, d = b.sample(sample_size)
            self.assert_np_array_equal(d,np.zeros((sample_size,1)))
            self.assert_np_array_equal(np.ones((sample_size,observation_space_size))*r,s0)
            self.assert_np_array_equal(np.ones((sample_size,observation_space_size))*r,s1)
            self.assert_np_array_equal(np.ones((sample_size,action_space_size))*r,a)

            indexes = r.reshape((sample_size,)).astype(int)

            # repeating indeces make it harder to use vectorized way
            for j in indexes:
                counters[j]+=1

        counters_sum = counters.sum()

        self.assertEqual(counters_sum,sample_count*sample_size)

        # avg number of times each index should have appeared in a random draw
        num_per_bin = counters_sum/size

        # check if within margins
        margin = 0.1
        for i in range(sample_size):
            self.assertTrue(counters[i]<num_per_bin*(1.+margin))
            self.assertTrue(counters[i]>num_per_bin*(1-margin))


    def assert_np_array_equal(self,a1,a2):
        self.assertTrue(np.square(a1-a2).sum() < 1e-3)


if __name__ == '__main__':
    unittest.main()
