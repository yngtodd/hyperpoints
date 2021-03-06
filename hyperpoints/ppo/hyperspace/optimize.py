import time
import argparse
import numpy as np

from objective import ppo_dist_objective
from hyperspace import hyperdrive
from hyperspace.kepler import load_results


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    start = time.time()

    space = [(2,10),
             (2,8),
             (2,6),
             (0.00, .99),
             (0.001, 0.1),
             (0.5, 0.99),
             (2.5e-4, 0.1),
             (0.5, 0.9)]

    checkpoint = load_results(args.results_dir)

    hyperdrive(objective=ppo_dist_objective,
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
