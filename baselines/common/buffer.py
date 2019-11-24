import numpy as np


class Buffer(object):

    def __init__(self, size, action_space_size, observation_space_size, reward_scaling = 1.):
        self.size = size
        self.current_size = 0
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.states_t0 = np.zeros((size, observation_space_size), dtype=np.float32)
        self.states_t1 = np.zeros((size, observation_space_size), dtype=np.float32)
        self.actions = np.zeros((size, action_space_size), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.index = 0
        self.reward_scaling = reward_scaling

    def add(self, st0, a, r, st1, d):

        self.states_t0[self.index, :] = st0
        self.states_t1[self.index, :] = st1
        self.actions[self.index, :] = a
        self.rewards[self.index, :] = r * self.reward_scaling
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