import tensorflow as tf
import numpy as np

import time

from baselines.common.actor_critic_mlps import ActorCriticMLPs
from baselines.common.utils import gaussian_likelihood, total_episode_reward_logger

from baselines.common import Buffer

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SAC_MLP_Networks(ActorCriticMLPs):

    def __init__(self, action_space_size, obs_space_size, layers, act_fun, layer_norm=False, target_network=False):
        if target_network:
            super().__init__(action_space_size, obs_space_size, layers, act_fun, layer_norm, create_actor=False,
                             qs_num=0, vs_num=1)
        else:
            super().__init__(action_space_size, obs_space_size, layers, act_fun, layer_norm, create_actor=True, qs_num=2,
                             vs_num=1)

    def create_actor_output(self, a_front):
        self._mu_layer = tf.keras.layers.Dense(
            self.action_space_size,
            dtype=tf.float32)

        self._mu_layer.build(input_shape=(self.layers[-1],))

        self._std_layer = tf.keras.layers.Dense(
            self.action_space_size,
            dtype=tf.float32)

        self._std_layer.build(input_shape=(self.layers[-1],))

        a_front.build()
        self.a_front = a_front

    def get_a_trainable_variables(self):
        return self.a_front.trainable_variables

    def get_q_trainable_variables(self, index):
        return self._qs[index].trainable_variables

    def get_v_trainable_variables(self):
        return self._vs[0].trainable_variables

    def get_interpolation_variables(self):
        return self.get_v_trainable_variables()

    @tf.function
    def get_a(self, states, training):
        common_input = self.a_front(states, training=training)

        mu = self._mu_layer(common_input, training=training)

        logstd = self._std_layer(common_input, training=training)
        logstd = tf.clip_by_value(logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(logstd)

        action = mu + tf.random.normal(shape=(self.action_space_size,)) * std

        log_likelihood = gaussian_likelihood(action, mu, logstd)

        squashed_mean_action = tf.tanh(mu)
        squashed_action = tf.tanh(action)
        # based on change of variables under squashing f(x) = tanh(x) which affects distribution
        # described under appendix C of https://arxiv.org/pdf/1812.05905.pdf
        squashed_log_likelihood = log_likelihood - tf.reduce_sum(tf.math.log(1. - squashed_action ** 2 + EPS), axis=1)

        return squashed_mean_action, squashed_action, squashed_log_likelihood


class Runner(object):
    def __init__(self, env, policy, buffer, writer=None, noise=None, learning_starts=100):
        self.env = env

        st0 = env.reset()

        self.st0 = st0.reshape((1, st0.size))
        self.policy = policy
        self.buffer = buffer
        self.writer = writer
        self.episode_reward = np.zeros((1,))
        self.num_timesteps = 0
        self.noise = noise
        self.learning_starts = learning_starts

    def run(self, rollout_steps):
        for i in range(rollout_steps):

            if self.num_timesteps < self.learning_starts:
                a_scaled = self.env.action_space.sample()
                a = self.undo_scale_action(a_scaled)
            else:
                _, a, _ = self.policy.get_a(self.st0, training=False)

                a = a.numpy().flatten()

                # add action noise
                if self.noise is not None:
                    a = self.noise.apply(a)
                    a = np.clip(a, -1, 1)

                a_scaled = self.scale_action(a)

            # true only for action spaces originally in tanh codomain
            # if np.sum(np.abs(a_scaled-a)) > 1e-3:
            #     print(a_scaled)
            #     print(a)
            #     assert False

            a = np.atleast_2d(a)
            a_scaled = np.atleast_2d(a_scaled)

            st1, reward, done, _ = self.env.step(a_scaled)

            if self.writer is not None:
                self.write_to_tensorboard(reward, done)

            self.buffer.add(self.st0, a, reward, st1, done)
            self.num_timesteps += 1

            self.st0 = st1.reshape(self.st0.shape)

    def scale_action(self, a):
        # normalize action from tanh codomain and denormalize to action space
        return (a + 1.) / 2 * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low

    def undo_scale_action(self, a):
        return (a - self.env.action_space.low)/(self.env.action_space.high - self.env.action_space.low)*2 - 1.

    def write_to_tensorboard(self, reward, done):
        ep_rew = np.array([reward]).reshape((1, -1))
        ep_done = np.array([done]).reshape((1, -1))
        self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                          self.writer, self.num_timesteps)


class SAC(object):
    def __init__(self,
                 env,
                 policy_kwargs,
                 nb_rollout_steps=1,
                 nb_train_steps=1,
                 batch_size=64,
                 buffer_size=50000,
                 learning_rate=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 action_noise=None,
                 layer_norm=False,
                 learning_starts=100
                 ):
        self.env = env
        self.policy_kwargs = policy_kwargs
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_train_steps = nb_train_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.learning_starts = learning_starts

        action_space_size = self.env.action_space.shape[0]
        observation_space_size = self.env.observation_space.shape[0]
        layers = self.policy_kwargs['layers']
        act_fun = self.policy_kwargs['act_fun']

        self.target_policy = SAC_MLP_Networks(action_space_size, observation_space_size, layers, act_fun, layer_norm,
                                              target_network=True)
        self.behavioral_policy = SAC_MLP_Networks(action_space_size, observation_space_size, layers, act_fun, layer_norm)

        self.buffer = Buffer(self.buffer_size, action_space_size, observation_space_size)

        writer = tf.summary.create_file_writer("./tensorboard/SAC_{}".format(time.time()))

        self.runner = Runner(self.env, self.behavioral_policy, self.buffer, writer, self.action_noise, learning_starts=learning_starts)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.entropy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # tf.config.experimental_run_functions_eagerly(True)
        self.target_policy.interpolate_variables(1., self.behavioral_policy)
        # tf.config.experimental_run_functions_eagerly(False)

        self.log_ent_coeff = tf.Variable(0., dtype=tf.float32, name='log_ent_coeff')
        self.ent_coeff = tf.exp(self.log_ent_coeff)
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)

    def learn(self, total_timesteps):
        current_rollout_steps = 0

        while current_rollout_steps < total_timesteps:

            self.runner.run(self.nb_rollout_steps)
            current_rollout_steps += self.nb_rollout_steps

            if self.buffer.can_sample(self.batch_size) and self.learning_starts < current_rollout_steps:
                for i in range(self.nb_train_steps):
                    data = self.buffer.sample(self.batch_size)
                    # data = tuple(tf.convert_to_tensor(d) for d in data)
                    self.train_step(*data)

    @tf.function
    def get_q_loss(self, states_t0, actions, rewards, states_t1, dones):

        # q losses

        not_dones = tf.ones_like(dones) - dones
        q_target = tf.stop_gradient(rewards + self.gamma * not_dones * self.target_policy.get_v(states_t1))

        q1 = self.behavioral_policy.get_q(states_t0, actions=actions, index=0)
        q2 = self.behavioral_policy.get_q(states_t0, actions=actions, index=1)

        bellman_error1 = q1 - q_target
        bellman_error2 = q2 - q_target

        q1_loss = 0.5 * tf.reduce_mean(tf.square(bellman_error1))
        q2_loss = 0.5 * tf.reduce_mean(tf.square(bellman_error2))

        return q1_loss + q2_loss

    @tf.function
    def get_a_loss(self, q1_pi, log_pi_a):
        # actor loss: actor is meant to maximize v_value = E[q_any - ent_coeff*log_pi]
        actor_loss = tf.negative(tf.reduce_mean(q1_pi - self.ent_coeff * log_pi_a))

        return actor_loss

    @tf.function
    def get_v_loss(self, states_t0, q1_pi, on_a, log_pi_a):

        q2_pi = self.behavioral_policy.get_q(states_t0, actions=on_a, index=1)

        q_min = tf.minimum(q1_pi, q2_pi)

        v_target = tf.stop_gradient(q_min - self.ent_coeff * log_pi_a)

        v = self.behavioral_policy.get_v(states_t0)

        v_loss = 0.5 * tf.reduce_mean(tf.square(v - v_target))

        return v_loss

    @tf.function
    def get_adaptive_entropy_loss(self,log_pi_a):
        # justified in https://arxiv.org/pdf/1812.05905.pdf in a rather involved way
        return -tf.reduce_mean(self.log_ent_coeff*tf.stop_gradient(log_pi_a+self.target_entropy))

    @tf.function
    def train_step(self, states_t0, actions, rewards, states_t1, dones):

        actor_variables = self.behavioral_policy.get_a_trainable_variables()

        critic_variables = \
            self.behavioral_policy.get_q_trainable_variables(0) + \
            self.behavioral_policy.get_q_trainable_variables(1) + \
            self.behavioral_policy.get_v_trainable_variables()

        with tf.GradientTape() as actor_tape:
            actor_tape.watch(actor_variables)
            _, on_a, log_pi_a = self.behavioral_policy.get_a(states_t0, training=False)
            q1_pi = self.behavioral_policy.get_q(states_t0, actions=on_a, index=0)
            actor_loss = self.get_a_loss(q1_pi, log_pi_a)

        actor_grad = actor_tape.gradient(actor_loss, actor_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_variables))

        with tf.GradientTape() as critic_tape:
            critic_tape.watch(critic_variables)
            critic_loss = self.get_q_loss(states_t0, actions, rewards, states_t1, dones) + \
            self.get_v_loss(states_t0, q1_pi, on_a, log_pi_a)

        critic_grad = critic_tape.gradient(critic_loss, critic_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic_variables))

        with tf.GradientTape() as entropy_tape:
            entropy_tape.watch(self.log_ent_coeff)
            entropy_loss = self.get_adaptive_entropy_loss(log_pi_a)

        entropy_grad = entropy_tape.gradient(entropy_loss,self.log_ent_coeff)

        self.entropy_optimizer.apply_gradients([(entropy_grad,self.log_ent_coeff)])

        self.target_policy.interpolate_variables(self.tau, self.behavioral_policy)
