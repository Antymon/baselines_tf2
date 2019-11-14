from ddpg2 import DDPG2
from ddpg2 import NormalNoise

import gym
import tensorflow as tf

if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')
    policy_kwargs = {'layers':[64,64],'act_fn':tf.keras.activations.tanh}

    alg = DDPG2(env, policy_kwargs, 500, 250, 1024, int(5e4), action_noise=NormalNoise(0.25))