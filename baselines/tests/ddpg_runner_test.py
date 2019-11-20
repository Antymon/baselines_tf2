import unittest
import gym
from baselines import MLPPolicy, Runner, Buffer, NormalNoise
import tensorflow as tf
import numpy as np

class DDPGRunnerTest(unittest.TestCase):
    def test_buffer_filling(self):
        env = gym.make('MountainCarContinuous-v0')
        policy_kwargs = {'layers': [64, 64], 'act_fn': tf.keras.activations.tanh}

        action_space_size = env.action_space.shape[0]
        observation_space_size = env.observation_space.shape[0]

        policy=MLPPolicy(action_space_size,observation_space_size,**policy_kwargs)

        size = 5
        b = Buffer(size, action_space_size, observation_space_size)


        run = Runner(env,policy,b)

        self.assertFalse(b.can_sample(2))
        run.run(2)
        self.assertTrue(b.can_sample(2))
    def test_action_scaling(self):

        class Env (object):
            def __init__(self):
                self.action_space = gym.spaces.Box(-15,5,(1,))
            def reset(self):
                return np.ones((1,1))

        run = Runner(Env(),None,None)

        self.assertAlmostEqual(run.scale_action(0.5),0,delta=1e-8)
        self.assertAlmostEqual(run.scale_action(-1),-15,delta=1e-8)
        self.assertAlmostEqual(run.scale_action(1),5,delta=1e-8)
        self.assertAlmostEqual(run.scale_action(0),-5,delta=1e-8)


if __name__ == '__main__':
    unittest.main()
