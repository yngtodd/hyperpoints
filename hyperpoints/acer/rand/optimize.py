import time
import argparse
import numpy as np

from skopt import dump
from skopt import dummy_minimize
from skopt.callbacks import CheckpointSaver

from objective import enduro_acer


def main():
    start = time.time()
    space = [(2,10),
             (2,8),
             (2,6),
             (0.00, .1),
             (0.1, 0.8),
             (1.0, 20.0),
             (0.0, 1.0)]

    checkpoint_saver = CheckpointSaver("./checkpoint.pkl")
    res_gp = dummy_minimize(enduro_acer, space, n_calls=20, random_state=0, callback=[checkpoint_saver], verbose=True)
    dump(res_gp, 'enduro_rand200.pkl')

    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
