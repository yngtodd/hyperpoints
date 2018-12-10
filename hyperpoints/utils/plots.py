import numpy as np
import seaborn as sns


def plot_rewards(rewards):
    """Plot aggregate results of series.

    Parameters
    ----------
    walks : pd.DataFrame
      Dataframe containing random walks as columns.
    """
    results = reward_stats(rewards)
    sns.set_context("paper")
    sns.tsplot(results)


def reward_stats(rewards):
    """Get mean and standard deviation of rewards.

    Parameters
    ----------
    walks : pd.DataFrame
      Dataframe containing random walks as columns.
    """
    mean = np.array(rewards.mean(axis=1))
    std_dev = np.array(rewards.std(axis=1))
    upper = mean + std_dev
    lower = mean - std_dev
    stats = [upper, mean, lower]
    return stats
