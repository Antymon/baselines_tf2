import unittest

import gym
import tensorflow as tf

import numpy as np

from baselines.ddpg import DDPG2
from baselines.common.noise import NormalNoise
from baselines.deps.vec_env.dummy_vec_env import DummyVecEnv


class DDPGTest(unittest.TestCase):
    def test_startup(self):
        env = gym.make('MountainCarContinuous-v0')
        policy_kwargs = {'layers': [4, 4], 'act_fun': tf.keras.activations.tanh}

        alg = DDPG2(env, policy_kwargs, 500, 250, 1024, int(5e4), action_noise=NormalNoise(0.25))

        sum_abs_diff = 0
        sum_abs = 0

        # after init networks should be identical
        for (var1,var2) in zip(alg.behavioral_policy._actor.trainable_variables+alg.behavioral_policy._critic.trainable_variables,
                                alg.target_policy._actor.trainable_variables+alg.target_policy._critic.trainable_variables
                               ):
            sum_abs_diff+=np.sum(np.abs(var1.numpy()-var2.numpy()))
            sum_abs+=np.sum(np.abs(var1.numpy()))

        self.assertAlmostEqual(sum_abs_diff,0,delta=alg.tau)

        # make sure that init was not with 0s, otherwise previous could have been meaningless
        self.assertNotAlmostEqual(sum_abs,0,delta=alg.tau)

    def test_learn(self):
        env = gym.make('MountainCarContinuous-v0')
        env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
        policy_kwargs = {'layers': [4, 4], 'act_fun': tf.keras.activations.tanh}

        alg = DDPG2(
            env,
            policy_kwargs,
            nb_rollout_steps=20,
            nb_train_steps=1,
            batch_size=5,
            buffer_size=10,
            action_noise=NormalNoise(0.25))

        cached_tensors=[]
        for t in (alg.target_policy._actor.trainable_variables+alg.target_policy._critic.trainable_variables):
            cached_tensors.append(t.numpy())

        alg.learn(1)

        sum_interpolated_diff = 0
        sum_identical_diff = 0

        # after one step weights should be interpolated
        for (var1,var2,cached_var2) in zip(
                alg.behavioral_policy._actor.trainable_variables+alg.behavioral_policy._critic.trainable_variables,
                alg.target_policy._actor.trainable_variables+alg.target_policy._critic.trainable_variables,
                cached_tensors
                               ):
            sum_interpolated_diff+=np.sum(np.abs(var2.numpy()-(1-alg.tau)*cached_var2-alg.tau*var1.numpy()))
            sum_identical_diff+=np.sum(np.abs(var1.numpy()-var2.numpy()))

        # after one step soft interpolation should have happened
        self.assertAlmostEqual(sum_interpolated_diff,0,delta=1e-3)

        # after one step weights should not be identical
        self.assertNotAlmostEqual(sum_identical_diff,0,delta=1e-3)

if __name__ == '__main__':
    unittest.main()
