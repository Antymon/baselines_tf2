import numpy as np
import tensorflow as tf

def total_episode_reward_logger(rew_acc, rewards, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    for env_idx in range(rewards.shape[0]):
        dones_idx = np.sort(np.argwhere(masks[env_idx]))

        if len(dones_idx) == 0:
            rew_acc[env_idx] += sum(rewards[env_idx])
        else:
            rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
            write_scalar(writer, rew_acc[env_idx], steps + dones_idx[0, 0])

            for k in range(1, len(dones_idx[:, 0])):
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k-1, 0]:dones_idx[k, 0]])
                write_scalar(writer, rew_acc[env_idx], steps + dones_idx[k, 0])

            rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc

# if plain scalars are sent to a tf.function new graph will be created each time vals are changed
# so they are packed here... to get unpacked later
# numpy arrays are accepted as proxy for tensors, i.e. their values do not create new cache entries
def write_scalar(writer, reward, step, tag="environment_info/episode_reward"):
    np_reward = np.atleast_1d(reward)
    np_step = np.atleast_1d(step)
    tf_write_scalar(tag, writer, np_reward, np_step)
    writer.flush()

@tf.function
def tf_write_scalar(tag, writer, value, step):
    with writer.as_default():
        # other model code would go here
        tf.summary.scalar(tag, value[0], step=step[0])

@tf.function
def gaussian_likelihood(input_, mu_, log_std, eps=1e-6):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.

    :param input_: (tf.Tensor)
    :param mu_: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: (tf.Tensor)
    """
    pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + eps)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)