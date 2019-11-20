import tensorflow as tf

from baselines.common.actor_critic_mlps import ActorCriticMLPs
from baselines.common.utils import gaussian_likelihood

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SAC_MLP_Networks(ActorCriticMLPs):

    def __init__(self, action_space_size, obs_space_size, layers, act_fn, layer_norm=False):
        super().__init__(action_space_size, obs_space_size, layers, act_fn, layer_norm, qs_num=2, vs_num=1)

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

    def get_interpolation_variables(self):
        return self._vs[0].trainable_variables

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
        squashed_log_likelihood = log_likelihood - tf.reduce_sum(tf.math.log(1. - squashed_action**2 + EPS), axis=1)

        return squashed_mean_action, squashed_action, squashed_log_likelihood
