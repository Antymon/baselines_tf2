import unittest
import gym
from baselines import Buffer
from baselines.sac import Runner,SAC_MLP_Networks
from baselines.deps.vec_env.dummy_vec_env import DummyVecEnv

import tensorflow as tf

class SACRunnerTest(unittest.TestCase):
    def test_buffer_filling(self):
        env = gym.make('MountainCarContinuous-v0')
        env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
        policy_kwargs = {'layers': [64, 64], 'act_fn': tf.keras.activations.tanh}

        action_space_size = env.action_space.shape[0]
        observation_space_size = env.observation_space.shape[0]

        policy=SAC_MLP_Networks(action_space_size,observation_space_size, **policy_kwargs)

        size = 5
        b = Buffer(size, action_space_size, observation_space_size)


        run = Runner(env,policy,b, learning_starts=0)

        self.assertFalse(b.can_sample(2))
        run.run(2)
        self.assertTrue(b.can_sample(2))

if __name__ == '__main__':
    unittest.main()
