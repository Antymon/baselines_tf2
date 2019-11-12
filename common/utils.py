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

            write_reward(writer, rew_acc[env_idx], steps + dones_idx[0, 0])
            writer.flush()

            for k in range(1, len(dones_idx[:, 0])):
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k-1, 0]:dones_idx[k, 0]])
                write_reward(writer, rew_acc[env_idx], steps + dones_idx[k, 0])

            rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc

@tf.function
def write_reward(writer, reward, step):
    with writer.as_default():
        # other model code would go here
        tf.summary.scalar("environment_info/episode_reward", reward, step=step)
