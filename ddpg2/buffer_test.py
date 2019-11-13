import unittest
from ddpg2 import Buffer

import numpy as np

class TestReplayBuffer(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
