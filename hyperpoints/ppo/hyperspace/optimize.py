import time
import argparse
import numpy as np
from hyperpoints.ppo.hyperspace import ppo_objective


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    start = time.time()

    space = [(2,10),
             (2,8),
             (3,8),
             (0.00, .99),
             (0.001, 0.1),
             (0.5, 0.99),
             (2.5e-4, 0.1),
             (0.5, 0.9)]

    hyperdrive(objective=ppo_objective,
               hyperparameters=space,
               results_path=args.results_dir,
               model="GP",
               n_iterations=100,
               verbose=True,
               random_state=0,
               checkpoints=True,
               restart=checkpoint)

    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
