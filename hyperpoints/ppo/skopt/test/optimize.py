import time
import numpy as np
from skopt import dump
from skopt import gp_minimize
from hyperpoints.ppo import ppo_objective
from skopt.space.space import Space, Integer, Real
import skopt


def main():
    start = time.time()

    print(f'Skopt version: {skopt.__version__}')

    space =  Space([Integer(low=2, high=7),
        Integer(low=5, high=8),
        Integer(low=5, high=6),
        Real(low=0.37124999999999997, high=0.99, prior='uniform', transform='identity'),
        Real(low=0.001, high=0.062875, prior='uniform', transform='identity'),
        Real(low=0.68375, high=0.99, prior='uniform', transform='identity'),
        Real(low=0.00025, high=0.06259375, prior='uniform', transform='identity'),
        Real(low=0.65, high=0.9, prior='uniform', transform='identity')])

    res_gp = gp_minimize(ppo_objective, space, n_calls=20, random_state=0, verbose=True)
    dump(res_gp, 'result200.pkl')
    print(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
