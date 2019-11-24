from baselines.sac import SAC

import gym
import tensorflow as tf

from baselines.deps.vec_env.vec_normalize import VecNormalize
from baselines.deps.vec_env.dummy_vec_env import DummyVecEnv

if __name__ == '__main__':

    env = gym.make('DartHexapod-v2')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_obs=10.)

    kwargs = dict(
        reward_scaling=1000., #only meaninggful non-default in this setting
        nb_rollout_steps=1,
        nb_train_steps=1,
        batch_size=64,
        buffer_size=int(5e5),
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        action_noise=None,
        layer_norm=False,
        learning_starts=int(1e4)
    )

    policy_kwargs = dict()
    policy_kwargs['act_fun'] = tf.nn.relu
    policy_kwargs['layers'] = [64,64]

    kwargs['policy_kwargs'] = policy_kwargs

    model = SAC(env, **kwargs)
    model.learn(total_timesteps=4e6)