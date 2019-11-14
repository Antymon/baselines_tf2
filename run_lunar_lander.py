from ddpg2.ddpg import DDPG2
from ddpg2.ddpg import NormalNoise



import gym
import tensorflow as tf
import time

from ddpg2.deps.vec_env.vec_normalize import VecNormalize
from ddpg2.deps.vec_env.dummy_vec_env import DummyVecEnv

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_obs=10.)

    kwargs = dict(
        nb_train_steps=100,
        nb_rollout_steps=500,
        tau=0.001,
        batch_size=2000,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=50000
    )

    policy_kwargs = dict()
    policy_kwargs['act_fn'] = tf.nn.relu
    policy_kwargs['layers'] = [400,200]

    kwargs['policy_kwargs'] = policy_kwargs

    stddev = 0.25
    kwargs['action_noise'] = NormalNoise(stddev)

    model = DDPG2(env, **kwargs)
    model.learn(total_timesteps=1e6)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        time.sleep(0.017)
