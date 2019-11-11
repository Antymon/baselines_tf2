import tensorflow as tf

tf.compat.v1.disable_eager_execution()

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


class Runner(object):
    def __init__(self):
        pass

class DDPG2(object):
    def __init__(self,
                 env,
                 policy_kwargs,
                 n_rollout_steps,
                 n_train_steps,
                 replay_size,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=0.001,
                 noise=None
                 ):
        self.env = env
        self.policy_kwargs = policy_kwargs
        self.n_rollout_steps = n_rollout_steps
        self.n_train_steps = n_train_steps
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
        self.actor_network = MLPPolicy(action_space_size,observation_space_size,layers,act_fn)



    def learn(self, total_timesteps):
        pass


    def get_losses(self,policy, observations):
        actor_loss =


    def train_step(self):
        pass