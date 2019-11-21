import unittest

import tensorflow as tf
import numpy as np

from baselines.sac import SAC_MLP_Networks, SAC

import gym
from baselines.common.noise import NormalNoise
from baselines.deps.vec_env.dummy_vec_env import DummyVecEnv


class SACTest(unittest.TestCase):
    def test_actor_output_shape(self):
        action_space_size = 5
        observation_space_size = 3
        sac_net = SAC_MLP_Networks(action_space_size,observation_space_size,[4,4],tf.tanh)

        state=np.random.normal(size=(1,observation_space_size))
        mean_action, action, _=sac_net.get_a(state,training=False)

        self.assertEqual((1,action_space_size),mean_action.shape)
        self.assertEqual((1,action_space_size),action.shape)

    def test_startup(self):
        env = gym.make('MountainCarContinuous-v0')
        policy_kwargs = {'layers': [4, 4], 'act_fun': tf.keras.activations.tanh}

        alg = SAC(env, policy_kwargs, action_noise=NormalNoise(0.25))

        sum_sq_diff = 0
        sum_sq = 0

        # after init networks should be identical
        for (var1,var2) in zip(alg.behavioral_policy.get_interpolation_variables(),
                                alg.target_policy.get_interpolation_variables()
                               ):
            sum_sq_diff+=np.sum(np.abs(var1.numpy()-var2.numpy()))
            sum_sq+=np.sum(np.abs(var1.numpy()))

        self.assertAlmostEqual(sum_sq_diff,0,delta=alg.tau)
        self.assertNotAlmostEqual(sum_sq,0,delta=alg.tau)

    def test_learn(self):
        env = gym.make('MountainCarContinuous-v0')
        env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
        policy_kwargs = {'layers': [4, 4], 'act_fun': tf.keras.activations.tanh}

        alg = SAC(env, policy_kwargs, action_noise=NormalNoise(0.25),learning_starts=0)

        cached_tensors=[]
        for t in (alg.target_policy.get_interpolation_variables()):
            cached_tensors.append(t.numpy())

        alg.learn(1)

        sum_sq_diff = 0
        sum_sq = 0

        # after one step weights should be intorpolated
        for (var1,var2,cached_var2) in zip(
                alg.behavioral_policy.get_interpolation_variables(),
                alg.target_policy.get_interpolation_variables(),
                cached_tensors
                               ):
            sum_sq_diff+=np.sum(np.abs(var2.numpy()-(1-alg.tau)*cached_var2-alg.tau*var1.numpy()))
            sum_sq+=np.sum(np.abs(var2.numpy()))

        self.assertAlmostEqual(sum_sq_diff,0,delta=alg.tau)
        self.assertNotAlmostEqual(sum_sq,0,delta=alg.tau)

if __name__ == '__main__':
    unittest.main()
