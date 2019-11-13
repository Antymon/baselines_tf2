import tensorflow as tf
import numpy as np
import time
from common.utils import total_episode_reward_logger

# tf.compat.v1.disable_eager_execution()

class Buffer(object):

    def __init__(self, size, action_space_size, observation_space_size):
        self.size = size
        self.current_size = 0
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.states_t0 = np.zeros((size, observation_space_size), dtype=np.float32)
        self.states_t1 = np.zeros((size, observation_space_size), dtype=np.float32)
        self.actions = np.zeros((size, action_space_size), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.index = 0

    def add(self, st0, a, r, st1, d):

        self.states_t0[self.index, :] = st0
        self.states_t1[self.index, :] = st1
        self.actions[self.index, :] = a
        self.rewards[self.index, :] = r
        self.dones[self.index, :] = 1.0 if d else 0

        self.index = (self.index + 1) % self.size
        self.current_size = min(self.size, self.current_size + 1)

    def can_sample(self, size):
        return self.current_size >= size

    def sample(self, size):
        if not self.can_sample(size):
            raise Exception('not enough records to sample')
        else:
            indeces = np.random.choice(self.current_size, size)
            return self.states_t0[indeces, :], \
                   self.actions[indeces, :], \
                   self.rewards[indeces, :], \
                   self.states_t1[indeces, :], \
                   self.dones[indeces, :]


class NormalNoise(object):
    def __init__(self, std):
        self.std = std

    def apply(self, noiseless_value):
        noisy_value = noiseless_value + np.random.normal(0, self.std, noiseless_value.shape)
        return noisy_value


class MLPPolicy(object):
    def __init__(self,
                 action_space_size,
                 obs_space_size,
                 layers,
                 act_fn):
        self._critic = None

        kwargs = dict(dtype=tf.float32)

        actor = tf.keras.Sequential()
        critic = tf.keras.Sequential()

        actor.add(tf.keras.layers.Dense(layers[0], input_shape=(obs_space_size,), **kwargs))
        critic.add(
            tf.keras.layers.Dense(layers[0], activation=act_fn, input_shape=(obs_space_size + action_space_size,),
                                  **kwargs))

        for i in range(1, len(layers)):
            actor.add(tf.keras.layers.Dense(layers[i], activation=act_fn, **kwargs))
            critic.add(tf.keras.layers.Dense(layers[i], activation=act_fn, **kwargs))

        actor.add(tf.keras.layers.Dense(action_space_size, activation=tf.keras.activations.tanh, **kwargs))
        critic.add(tf.keras.layers.Dense(1, **kwargs))

        actor.build()
        critic.build()

        self._actor = actor
        self._critic = critic

    @tf.function
    def get_a(self, state, training):
        return self._actor(state, training=training)

    @tf.function
    def get_q(self, states, actions=None):
        if actions is None:
            actions = self.get_a(states, training=True)

        q_input = tf.concat([states, actions], -1)
        qs = self._critic(q_input, training=True)

        return qs

    def get_trainable_variables(self):
        return self._actor.trainable_variables + self._critic.trainable_variables

    @tf.function
    def update_trainable_variables(self, tau, other_policy):

        other_variables = other_policy.get_trainable_variables()
        self_variables = self.get_trainable_variables()

        for (self_var, other_var) in zip(self_variables, other_variables):
            self_var.assign((1. - tau) * self_var + tau * other_var)


class Runner(object):
    def __init__(self, env, policy, buffer, writer=None, noise=None):
        self.env = env

        st0 = env.reset()

        self.st0 = st0.reshape((1, st0.shape[0]))
        self.policy = policy
        self.buffer = buffer
        self.writer = writer
        self.episode_reward = np.zeros((1,))
        self.num_timesteps = 0
        self.noise = noise

    def run(self, rollout_steps):
        for i in range(rollout_steps):
            a = self.policy.get_a(self.st0, training=False)

            a = a.numpy().flatten()

            # add action noise
            if self.noise is not None:
                a = self.noise.apply(a)
                a = np.clip(a, -1, 1)

            a = self.scale_action(a)

            st1, reward, done, _ = self.env.step(a)

            if self.writer is not None:
                self.write_to_tensorboard(reward, done)

            self.buffer.add(self.st0, a, reward, st1, done)
            self.num_timesteps += 1

            self.st0 = st1.reshape(self.st0.shape)

    def scale_action(self, a):
        # normalize action from tanh codomain and denormalize to action space
        return (a + 1.) / 2 * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low

    def write_to_tensorboard(self, reward, done):
        ep_rew = np.array([reward]).reshape((1, -1))
        ep_done = np.array([done]).reshape((1, -1))
        self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                          self.writer, self.num_timesteps)


class DDPG2(object):
    def __init__(self,
                 env,
                 policy_kwargs,
                 n_rollout_steps,
                 n_train_steps,
                 batch_size,
                 replay_size,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=0.001,
                 noise=None,

                 ):
        self.env = env
        self.policy_kwargs = policy_kwargs
        self.n_rollout_steps = n_rollout_steps
        self.n_train_steps = n_train_steps
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.noise = noise

        #     self.setup_model()
        #
        # def setup_model(self):

        action_space_size = self.env.action_space.shape[0]
        observation_space_size = self.env.observation_space.shape[0]
        layers = self.policy_kwargs['layers']
        act_fn = self.policy_kwargs['act_fn']

        self.target_policy = MLPPolicy(action_space_size, observation_space_size, layers, act_fn)
        self.behavioral_policy = MLPPolicy(action_space_size, observation_space_size, layers, act_fn)

        self.buffer = Buffer(self.replay_size, action_space_size, observation_space_size)

        writer = tf.summary.create_file_writer("./tensorboard/DDPG_{}".format(time.time()))

        self.runner = Runner(self.env, self.behavioral_policy, self.buffer, writer, self.noise)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

        # tf.config.experimental_run_functions_eagerly(True)
        self.target_policy.update_trainable_variables(1., self.behavioral_policy)
        # tf.config.experimental_run_functions_eagerly(False)

    def learn(self, total_timesteps):
        current_rollout_steps = 0

        while current_rollout_steps < total_timesteps:

            self.runner.run(self.n_rollout_steps)
            current_rollout_steps += self.n_rollout_steps

            if self.buffer.can_sample(self.batch_size):
                for i in range(self.n_train_steps):
                    data = self.buffer.sample(self.batch_size)
                    # data = tuple(tf.convert_to_tensor(d) for d in data)
                    self.train_step(*data)

    @tf.function
    def get_actor_loss(self, states_t0):

        # actor is meant to maximize q_value

        qs = self.behavioral_policy.get_q(states_t0)
        actor_loss = tf.negative(tf.reduce_mean(qs))
        return actor_loss

    @tf.function
    def get_critic_loss(self, states_t0, actions, rewards, states_t1, dones):
        # critic is meant to minimize Bellman error

        not_dones = tf.ones_like(dones) - dones
        targets = rewards + self.gamma * not_dones * self.target_policy.get_q(states_t1)
        bellman_error = self.behavioral_policy.get_q(states_t0, actions=actions) - targets
        critic_loss = tf.reduce_mean(tf.square(bellman_error))

        return critic_loss

    @tf.function
    def train_step(self, states_t0, actions, rewards, states_t1, dones):

        actor_variables = self.behavioral_policy._actor.trainable_variables
        critic_variables = self.behavioral_policy._critic.trainable_variables

        with tf.GradientTape() as actor_tape:
            actor_tape.watch(actor_variables)
            actor_loss = self.get_actor_loss(states_t0)

        actor_grad = actor_tape.gradient(actor_loss, actor_variables)

        with tf.GradientTape() as critic_tape:
            critic_tape.watch(critic_variables)
            critic_loss = self.get_critic_loss(states_t0, actions, rewards, states_t1, dones)

        critic_grad = critic_tape.gradient(critic_loss, critic_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic_variables))

        self.target_policy.update_trainable_variables(self.tau, self.behavioral_policy)
