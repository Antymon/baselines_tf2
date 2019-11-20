import tensorflow as tf

from baselines.common.actor_critic_mlps import ActorCriticMLPs

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
        common_input = self.a_front(states,training=training)
        mu=self._mu_layer(common_input,training=training)
        std=self._std_layer(common_input,training=training)
        sample = mu+tf.random.normal(shape=(self.action_space_size,))*std
        return tf.tanh(sample)