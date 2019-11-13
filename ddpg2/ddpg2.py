import tensorflow as tf
import numpy as np
import time
from common.utils import total_episode_reward_logger

# tf.compat.v1.disable_eager_execution()

class Buffer(object):

    def __init__(self, size, action_space_size, observation_space_size):
        self.size = size
        self.current_size = 0
        self.rewards = np.zeros((size,1))
        self.states_t0 = np.zeros((size,observation_space_size))
        self.states_t1 = np.zeros((size,observation_space_size))
        self.actions = np.zeros((size,action_space_size))
        self.dones = np.zeros((size,1))
        self.index = 0

    def add(self, record):
        st0,a,r,st1,d = record

        self.states_t0[self.index,:] = st0
        self.states_t1[self.index,:] = st1
        self.actions[self.index,:] = a
        self.rewards[self.index,:] = r
        self.dones[self.index,:] = 1.0 if d else 0

        self.index = (self.index+1) % self.size
        self.current_size = min(self.size,self.current_size+1)

    def can_sample(self,size):
        return self.current_size > size

    def sample(self,size):
        if not self.can_sample(size):
            raise Exception('not enough records to sample')
        else:
            indeces = np.random.choice(self.current_size, size)
            return self.states_t0[indeces,:], \
                   self.actions[indeces,:], \
                   self.rewards[indeces,:], \
                   self.states_t1[indeces,:], \
                   self.dones[indeces,:]

class NormalNoise(object):
    def __init__(self,std):
        self.std = std

    def apply(self,noiseless_value):
        noisy_value = noiseless_value+np.random.normal(0,self.std,noiseless_value.size())
        return noisy_value

class MLPPolicy(object):
    def __init__(self,
                 action_space_size,
                 obs_space_size,
                 layers,
                 act_fn):
        self.critic = None

        actor = tf.keras.Sequential()
        critic = tf.keras.Sequential()

        actor.add(tf.keras.layers.Dense(layers[0], activation=act_fn, input_shape=(obs_space_size,)))
        critic.add(tf.keras.layers.Dense(layers[0], activation=act_fn, input_shape=(obs_space_size+action_space_size,)))

        for i in range(1,len(layers)):
            actor.add(tf.keras.layers.Dense(layers[i], activation=act_fn))
            critic.add(tf.keras.layers.Dense(layers[i], activation=act_fn))

        actor.add(tf.keras.layers.Dense(action_space_size, activation=tf.keras.activations.tanh))
        critic.add(tf.keras.layers.Dense(1))

        actor.build()
        critic.build()

        self.actor = actor
        self.critic = critic

    def get_a(self, state, training):
        return self.actor(state, training=training)

    def get_q(self, states, actions=None):
        if actions is None:
            actions = self.get_a(states, training=True)

        q_input = tf.concat([states,actions],-1)
        qs = self.critic(q_input, training=True)

        return qs

    @tf.function
    def update_trainable_variables(self,tau,other_policy):

        other_variables = other_policy.actor.trainable_variables + other_policy.critic.trainable_variables
        self_variables = self.actor.trainable_variables + self.critic.trainable_variables

        for (self_var,other_var) in zip(self_variables,other_variables):
            self_var.assign((1. - tau)*self_var+tau*other_var)



class Runner(object):
    def __init__(self, env, policy, rollout_steps, buffer,writer, noise):
        self.rollout_steps = rollout_steps
        self.env = env
        self.s = env.reset()
        self.policy = policy
        self.buffer = buffer
        self.writer = writer
        self.episode_reward = np.zeros((1,))
        self.num_timesteps = 0
        self.noise = noise

    def run(self):
        for i in range(self.rollout_steps):
            a = self.policy.get_a(self.s, training=False)

            a = a.flatten()

            # add action noise
            if self.noise is not None:
                a = self.noise.apply(a)
                a = np.clip(a, -1, 1)

            # normalize action from tanh codomain and denormalize to action space
            a = (a+1.)/2*(self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low

            record = self.env.step(a)

            if self.writer is not None:
                _, reward, done, _ = record
                self.write_to_tensorboard(reward,done)

            # skipping additional info at the end
            self.buffer.add(record[:-1])
            self.num_timesteps += 1

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

        self.setup_model()

    def setup_model(self):

        action_space_size = self.env.action_space.shape[0]
        observation_space_size = self.env.observation_space.shape[0]
        layers = self.policy_kwargs['layers']
        act_fn = self.policy_kwargs['act_fn']

        self.target_network = MLPPolicy(action_space_size,observation_space_size,layers,act_fn)
        self.behavioral_network = MLPPolicy(action_space_size, observation_space_size, layers, act_fn)

        self.buffer = Buffer(self.replay_size,action_space_size,observation_space_size)

        writer = tf.summary.create_file_writer("./tensorboard/DDPG_{}".format(time.time()))

        self.runner = Runner(self.env,self.behavioral_network.actor,self.n_rollout_steps,self.buffer, writer, self.noise)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

        # tf.config.experimental_run_functions_eagerly(True)
        self.target_network.update_trainable_variables(1.,self.behavioral_network)
        # tf.config.experimental_run_functions_eagerly(False)

    def learn(self, total_timesteps):
        rollout_steps=0

        while rollout_steps < total_timesteps:

            self.runner.run()
            rollout_steps+=rollout_steps

            if self.buffer.can_sample(self.replay_size):
                for i in range(self.train_step):
                    data = self.buffer.sample(self.batch_size)
                    tensor_data = tuple(tf.convert_to_tensor(d) for d in data)
                    self.train_step(*tensor_data)


    def get_losses(self, states_t0, actions, rewards, states_t1, dones):

        # actor is meant to maximize q_value

        qs = self.behavioral_policy.get_q(states_t0)
        actor_loss = tf.negative(tf.reduce_mean(qs))

        # critic is meant to minimize Bellman error

        not_dones = tf.ones_like(dones)-dones
        targets = rewards + self.gamma*not_dones*self.target_policy.get_q(states_t1)
        bellman_error = self.behavioral_policy.get_q(states_t0,actions=actions)-targets
        critic_loss = tf.reduce_mean(tf.square(bellman_error))

        return actor_loss, critic_loss


    @tf.function
    def train_step(self, states_t0, actions, rewards, states_t1, dones):

        actor_variables = self.behavioral_network.actor.trainable_variables
        critic_variables = self.behavioral_network.critic.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(actor_variables)
            tape.watch(critic_variables)
            actor_loss, critic_loss = self.get_losses(states_t0, actions, rewards, states_t1, dones)

        actor_grad = tape.Gradient(actor_loss,actor_variables)
        critic_grad = tape.Gradient(critic_loss,critic_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grad,actor_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grad,critic_variables))

        self.target_network.update_trainable_variables(self.tau,self.behavioral_network)