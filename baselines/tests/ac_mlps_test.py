import unittest

import tensorflow as tf

from baselines.common.actor_critic_mlps import ActorCriticMLPs

class DummyNetworks(ActorCriticMLPs):

    def create_actor_output(self, a_front):
        pass

    def get_interpolation_variables(self):
        pass

    @tf.function
    def get_a(self, states, training):
        pass

# class MockedNetworks(ActorCriticMLPs):
#
#     def create_actor_output(self, a_front):
#         a_front.add(tf.keras.layers.Dense(
#             self.action_space_size,
#             activation=tf.keras.activations.tanh,
#             dtype=tf.float32))
#
#         self._a = a_front
#
#     def get_interpolation_variables(self):
#         return self._vs[0].trainable_variables
#
#     @tf.function
#     def get_a(self, states, training):
#         pass


class AC_MLPs_Test(unittest.TestCase):
    def test_cant_instantiate(self):
        try:
            nets = ActorCriticMLPs(2,3,[4,4],tf.tanh,False,2,1)
        except TypeError:
            pass
        except:
            self.assertTrue(False,"Expected TypeError due to abstract class instantiation but different occurred")
        else:
            self.assertTrue(False,"Expected TypeError due to abstract class instantiation but none occurred")

    # def test_dummy_not_creating_actor_fails(self):
    #     try:
    #         nets = DummyNetworks(2, 3, [4, 4],tf.tanh, False, 2, 1)
    #     except AssertionError as e:
    #         pass
    #     except:
    #         self.assertTrue(False,"Expected AssertionError but different occurred")
    #     else:
    #         self.assertTrue(False,"Expected AssertionError but none occurred")

    def test_successful_creation(self):

        num_qs=2
        num_vs=1

        nets = DummyNetworks(2, 3, [4, 4], tf.tanh, False,False, num_qs, num_vs)

        self.assertEqual(len(nets._qs),num_qs)
        self.assertEqual(len(nets._vs),num_vs)

        q=nets._qs[0]

        for layer in q.layers:
            print(layer.output_shape)

        print(q.summary())




if __name__ == '__main__':
    unittest.main()
