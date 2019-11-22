from baselines.sac import SAC
from baselines import NormalNoise

import gym
import tensorflow as tf
import time

from baselines.deps.vec_env.vec_normalize import VecNormalize
from baselines.deps.vec_env.dummy_vec_env import DummyVecEnv

if __name__ == '__main__':

    env = gym.make('DartHexapod-v2')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_obs=10.)

    kwargs = dict(
        nb_rollout_steps=1,
        nb_train_steps=1,
        batch_size=512,
        buffer_size=500000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        action_noise=None,
        layer_norm=False,
        learning_starts=10000
    )

    policy_kwargs = dict()
    policy_kwargs['act_fun'] = tf.nn.tanh
    policy_kwargs['layers'] = [6,6]

    kwargs['policy_kwargs'] = policy_kwargs

    # stddev = 0.25
    # kwargs['action_noise'] = NormalNoise(stddev)

    model = SAC(env, **kwargs)
    model.learn(total_timesteps=2e7)

    obs = env.reset()
    reward = 0
    while(True):
        action = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward+=rewards[0]
        env.render()
        time.sleep(0.017)
        if dones[0]:
            print("eval ep rew: {}".format(reward))
            reward=0
