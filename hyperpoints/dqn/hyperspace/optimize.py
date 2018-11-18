import time
import argparse
import numpy as np

from objective import enduro_dist_dqn
from hyperspace import hyperdrive


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    start = time.time()

    space = [(2,10),
             (2,8),
             (3,6),
             (0.00, .99),
             (50000, 250000),
             (8, 32)]

    hyperdrive(objective=enduro_dist_dqn,
               hyperparameters=space,
               results_path=args.results_dir,
               model="GP",
               n_iterations=20,
               verbose=True,
               random_state=0,
               checkpoints=True)

    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()

