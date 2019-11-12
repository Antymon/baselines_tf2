import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


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
    def __init__(self):
        pass

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

    def get_q(self, states, actions=None):
        if actions is None:
            actions = self.actor(states)

        q_input = tf.concat([states,actions],-1)
        qs = self.critic(q_input)

        return qs

    def update_trainable_variables(self,tau,other_policy):

        other_variables = other_policy.actor.trainable_variables + other_policy.critic.trainable_variables
        self_variables = self.actor.trainable_variables + self.critic.trainable_variables

        for (self_var,other_var) in zip(self_variables,other_variables):
            self_var.assign((1. - tau)*self_var+tau*other_var)



class Runner(object):
    def __init__(self, env, actor, rollout_steps, buffer):
        self.rollout_steps = rollout_steps
        self.env = env
        self.s = env.reset()
        self.actor = actor
        self.buffer = buffer

    def run(self):
        for i in range(self.rollout_steps):
            a = self.actor(self.s)
            # add action noise
            record = self.env.step(a)
            # skipping additional info at the end
            self.buffer.add(record[:-1])

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
        self.runner = Runner(self.env,self.n_rollout_steps,self.buffer)

        self.actor_optimizer = tf.keras.optimizers.Adam()
        self.critic_optimizer = tf.keras.optimizers.Adam()

        self.init_networks()

    @tf.function
    def init_networks(self):
        self.target_network.update_trainable_variables(1.,self.behavioral_network)

    def learn(self, total_timesteps):

        rollout_steps=0

        while rollout_steps < total_timesteps:

            self.runner.run()
            rollout_steps+=rollout_steps

            if self.buffer.can_sample(self.replay_size):
                for i in range(self.train_step):
                    self.train_step()


    def get_losses(self,behavioral_policy,target_policy):

        states_t0, actions, rewards, states_t1, dones = self.buffer.sample(self.batch_size)

        # actor is meant to maximize q_value

        qs = behavioral_policy.get_q(states_t0)
        actor_loss = tf.negative(tf.reduce_mean(qs))

        # critic is meant to minimize Bellman error

        not_dones = tf.ones_like(dones)-dones
        targets = rewards + self.gamma*not_dones*target_policy.get_q(states_t1)
        bellman_error = behavioral_policy.get_q(states_t0,actions)-targets
        critic_loss = tf.reduce_mean(tf.square(bellman_error))

        return actor_loss, critic_loss


    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            actor_loss, critic_loss = self.get_losses(self.behavioral_network,self.target_network)

        actor_variables = self.behavioral_network.actor.trainable_variables
        critic_variables = self.behavioral_network.critic.trainable_variables

        actor_grad = tape.Gradient(actor_loss,actor_variables)
        critic_grad = tape.Gradient(critic_loss,critic_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grad,actor_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grad,critic_variables))

        self.target_network.update_trainable_variables(self.tau,self.behavioral_network)
            