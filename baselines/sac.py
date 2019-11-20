class SAC_MLP_Networks(ActorCriticMLPs):

    def __init__(self, action_space_size, obs_space_size, layers, act_fn, layer_norm=False, qs_num=1, vs_num=0):
        super().__init__(action_space_size, obs_space_size, layers, act_fn, layer_norm, qs_num, vs_num)

    def create_actor_output(self, a_front):
        self._mu_layer = tf.keras.layers.Dense(
            self.action_space_size,
            dtype=tf.float32)

        self._std_layer = tf.keras.layers.Dense(
            self.action_space_size,
            dtype=tf.float32)

        auxiliary_input = tf.keras.layers.Input(shape=(self.obs_space_size,), name='aux_input')

        common_input = a_front(auxiliary_input)
        mu=self._mu_layer(common_input)
        std=self._std_layer(common_input)

        std.build()

        self._a = mu+tf.random.normal(shape=(self.action_space_size,))*std

    def get_interpolation_variables(self):
        return self._vs[0].trainable_variables