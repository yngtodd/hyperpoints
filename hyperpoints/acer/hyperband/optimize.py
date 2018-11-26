import time
import numpy as np
from skopt import dump
from hyperspace import hyperband
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

    res = hyperband(enduro_acer, space, n_evaluations=20, max_iter=200, eta=3, verbose=True, random_state=0)
    dump(res, 'hyperband_enduro200.pkl')
    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
