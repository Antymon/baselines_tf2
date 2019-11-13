import unittest

from ddpg2 import DDPG2
from ddpg2 import NormalNoise

import gym
import tensorflow as tf

import numpy as np

class AlgorithmTest(unittest.TestCase):
    def test_startup(self):
        env = gym.make('MountainCarContinuous-v0')
        policy_kwargs = {'layers': [4, 4], 'act_fn': tf.keras.activations.tanh}

        alg = DDPG2(env, policy_kwargs, 500, 250, 1024, int(5e4), noise=NormalNoise(0.25))

        sum_sq_diff = 0
        sum_sq = 0

        # after init networks should be identical
        for (var1,var2) in zip(alg.behavioral_policy._actor.trainable_variables+alg.behavioral_policy._critic.trainable_variables,
                                alg.target_policy._actor.trainable_variables+alg.target_policy._critic.trainable_variables
                               ):
            sum_sq_diff+=np.sum(np.abs(var1.numpy()-var2.numpy()))
            sum_sq+=np.sum(np.abs(var1.numpy()))

        self.assertAlmostEqual(sum_sq_diff,0,delta=alg.tau)
        self.assertNotAlmostEqual(sum_sq,0,delta=alg.tau)

    def test_learn(self):
        env = gym.make('MountainCarContinuous-v0')
        policy_kwargs = {'layers': [4, 4], 'act_fn': tf.keras.activations.tanh}

        alg = DDPG2(
            env,
            policy_kwargs,
            nb_rollout_steps=20,
            nb_train_steps=1,
            batch_size=5,
            buffer_size=10,
            noise=NormalNoise(0.25))

        cached_tensors=[]
        for t in (alg.target_policy._actor.trainable_variables+alg.target_policy._critic.trainable_variables):
            cached_tensors.append(t.numpy())

        alg.learn(1)

        sum_sq_diff = 0
        sum_sq = 0

        # after one step weights should be intorpolated
        for (var1,var2,cached_var2) in zip(
                alg.behavioral_policy._actor.trainable_variables+alg.behavioral_policy._critic.trainable_variables,
                alg.target_policy._actor.trainable_variables+alg.target_policy._critic.trainable_variables,
                cached_tensors
                               ):
            sum_sq_diff+=np.sum(np.abs(var2.numpy()-(1-alg.tau)*cached_var2-alg.tau*var1.numpy()))
            sum_sq+=np.sum(np.abs(var2.numpy()))

        self.assertAlmostEqual(sum_sq_diff,0,delta=alg.tau)
        self.assertNotAlmostEqual(sum_sq,0,delta=alg.tau)


if __name__ == '__main__':
    unittest.main()
