import numpy as np
import pandas as pd
from pathlib import Path


def gather_rewards(path: str, savefile: str=None) -> pd.DataFrame:
    """Collects results from multiple test runs.

    Paramters
    ---------
    path : str
      Path to vel.output.openai results for experiment.

    savefile : str [optional]
      Name to save concatenated rewards to.
    """
    p = Path(path)
    results_paths = [x for x in p.iterdir() if x.is_dir()]
    results_paths = [x.joinpath('progress.csv') for x in results_paths]

    results = []
    for i in results_paths:
        try:
            res = pd.read_csv(i)
            if len(res) > 1000:
                results.append(res)
        except Exception:
            pass

    print(f'Number of results: {len(results)}')
    rewards = [np.array(x['PMM:episode_rewards']) for x in results]
    rewards = pd.concat([pd.Series(x) for x in rewards], axis=1)

    if savefile:
        rewards.to_csv(savefile, index=False)

    return rewards
