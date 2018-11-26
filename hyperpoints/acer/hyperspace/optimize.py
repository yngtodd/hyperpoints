import time
import argparse
import numpy as np

from objective import enduro_acer
from hyperspace import hyperdrive


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    start = time.time()

    space = [(2,10),
             (2,8),
             (2,6),
             (0.00, .1),
             (0.1, 0.8),
             (1.0, 20.0),
             (0.0, 1.0)]

    hyperdrive(objective=enduro_acer,
               hyperparameters=space,
               results_path=args.results_dir,
               checkpoints_path=args.results_dir,
               model="GP",
               n_iterations=20,
               verbose=True,
               random_state=0)

    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
