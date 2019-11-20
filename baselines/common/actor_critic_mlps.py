import tensorflow as tf
from abc import ABC, abstractmethod

class ActorCriticMLPs(ABC):
    def __init__(self,
                 action_space_size,
                 obs_space_size,
                 layers,
                 act_fn,
                 layer_norm=False,
                 qs_num=1,
                 vs_num=0):
        self.action_space_size = action_space_size
        self.obs_space_size = obs_space_size
        self.layers = layers
        self.act_fn = act_fn
        self.layer_norm = layer_norm

        self._a = None
        self._qs = []
        self._vs = []

        a_front = self.create_mlp(input_shape=(obs_space_size,))

        self.create_actor_output(a_front)

        for _ in range(qs_num):
            self._qs.append(self.create_mlp(input_shape=(obs_space_size + action_space_size,),output_size=1))

        for _ in range(vs_num):
            self._vs.append(self.create_mlp(input_shape=(obs_space_size,), output_size=1))

        for model in self._qs+self._vs:
            model.build()

    @abstractmethod
    def create_actor_output(self, a_front):
        raise NotImplementedError()

    def create_mlp(self, input_shape, output_size = None):
        model=tf.keras.Sequential()

        self.add_segment(model,self.layers[0],input_shape=input_shape)

        for i in range(1, len(self.layers)):
            self.add_segment(model,self.layers[i])

        if output_size is not None:
            self.add_segment(model, output_size, add_act=False)

        return model

    def add_segment(self, network, output_shape, add_act=True, **kwargs):
        network.add(tf.keras.layers.Dense(output_shape,dtype=tf.float32,**kwargs))
        if self.layer_norm:
            network.add(tf.keras.layers.LayerNormalization(center=True, scale=True))
        if add_act:
            network.add(tf.keras.layers.Activation(self.act_fn))

    @abstractmethod
    @tf.function
    def get_a(self, states, training):
        raise NotImplementedError()

    @tf.function
    def get_q(self, states, actions=None, index=0):
        if actions is None:
            actions = self.get_a(states, training=True)

        q_input = tf.concat([states, actions], -1)
        q_vals = self._qs[index](q_input, training=True)

        return q_vals

    @tf.function
    def get_v(self, states, index=0):

        q_input = tf.concat([states], -1)
        q_vals = self._vs[index](q_input, training=True)

        return q_vals

    @abstractmethod
    def get_interpolation_variables(self):
        raise NotImplementedError()

    @tf.function
    def interpolate_variables(self, tau, other_policy):

        other_variables = other_policy.get_interpolation_variables()
        self_variables = self.get_interpolation_variables()

        for (self_var, other_var) in zip(self_variables, other_variables):
            self_var.assign((1. - tau) * self_var + tau * other_var)