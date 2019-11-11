from ddpg2 import DDPG2

import gym
import tensorflow as tf

if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')
    policy_kwargs = {'layers':[64,64],'act_fn':tf.keras.activations.tanh}

    alg = DDPG2(env,policy_kwargs,-1,-1,-1)