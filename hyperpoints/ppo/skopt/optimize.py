import time
import numpy as np
from skopt import dump
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from hyperpoints.ppo import ppo_objective


def main():
    start = time.time()
    checkpoint_saver = CheckpointSaver("./checkpoint.pkl")
    space = [(2,10), (2,8), (3,8), (0.00, .99), (0.001, 0.1), (0.5, 0.99), (2.5e-4, 0.1), (0.5, 0.9)]
    res_gp = gp_minimize(ppo_objective, space, n_calls=20, random_state=0, callback=[checkpoint_saver], verbose=True)
    dump(res_gp, 'result200.pkl')
    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
