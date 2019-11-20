import unittest

import tensorflow as tf
import numpy as np

from baselines.sac import SAC_MLP_Networks


class SACTest(unittest.TestCase):
    def test_actor_output_shape(self):
        action_space_size = 5
        observation_space_size = 3
        sac_net = SAC_MLP_Networks(action_space_size,observation_space_size,[4,4],tf.tanh)

        state=np.random.normal(size=(1,observation_space_size))
        action=sac_net.get_a(state,training=False)

        self.assertEqual((1,action_space_size),action.shape)



if __name__ == '__main__':
    unittest.main()
