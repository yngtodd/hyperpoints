import time
import numpy as np
from skopt import dump
from hyperspace import hyperband
from hyperpoints.ppo.hyperband import hyperband_objective


def main():
    start = time.time()
    space = [(2,10), (2,8), (3,6), (0.00, .99), (0.001, 0.1), (0.5, 0.99), (2.5e-4, 0.1), (0.5, 0.9)]
    res = hyperband(hyperband_objective, space, n_evaluations=20, max_iter=200, eta=3, verbose=True, random_state=0)
    dump(res, 'result200.pkl')
    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
