import unittest

from ddpg2 import DDPG2
from ddpg2 import NormalNoise

import gym
import tensorflow as tf

import numpy as np

class MyTestCase(unittest.TestCase):
    def test_startup(self):
        env = gym.make('MountainCarContinuous-v0')
        policy_kwargs = {'layers': [4, 4], 'act_fn': tf.keras.activations.tanh}

        alg = DDPG2(env, policy_kwargs, 500, 250, 1024, int(5e4), noise=NormalNoise(0.25))

        sum_sq_diff = 0
        sum_sq = 0

        for (var1,var2) in zip(alg.behavioral_network._actor.trainable_variables+alg.behavioral_network._critic.trainable_variables,
                                alg.target_network._actor.trainable_variables+alg.target_network._critic.trainable_variables
                               ):
            sum_sq_diff+=np.sum(np.square(var1.numpy()-var2.numpy()))
            sum_sq+=np.sum(np.square(var1.numpy()))
            # print(var2.numpy())

        self.assertAlmostEqual(sum_sq_diff,0,delta=1e-3)
        self.assertNotAlmostEqual(sum_sq,0,delta=1e-3)

if __name__ == '__main__':
    unittest.main()
