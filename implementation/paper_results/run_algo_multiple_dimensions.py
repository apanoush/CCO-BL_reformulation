import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
from tqdm import tqdm
from optimizer import Optimizer
import multiprocessing
from taco_paper_problem import TacoPaperProblem1, TacoPaperProblem2
import pickle
import json
from algorithm import Algorithm
from oracle import Oracle
from problem import Problem


def run_algorithm(args):

    problem, ZERO_ORDER_METHOD = args
    
    algorithm = Algorithm(problem, verbose=True, empirical_coverage=False, plots=False, zero_order_method=ZERO_ORDER_METHOD)
    opt = Optimizer(algorithm)    
    opt.run()
    # empirical_coverage = opt.oracle.empirical_coverage(opt.result)
    # f_value = problem.f_function(opt.result)

    suboptimality = Oracle.suboptimality_f(problem, opt.algorithm.x_history)

    return [problem.dimension, suboptimality, algorithm.chance_constraint_quantile_history, algorithm.x_history]

def main(ZERO_ORDER_METHOD):

    INITIAL_X = [[0.1] * d for d in [2, 10, 50, 200]]

    args = []
    for initial_x in INITIAL_X:
        problem = TacoPaperProblem2(
            initial_x=initial_x, N=10, T=300, I=300, GA_steps=500, max_iters=1200, theta_G=1-0.8, theta_H=0.9, lambda_=0, learning_rate_SGLD=6e-1, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=2e1, delta= 7e-1, K=1e-2
        )
        args.append((problem, ZERO_ORDER_METHOD))

    filename = f"multiple_dimensions_{type(problem).__name__}.pkl" if not ZERO_ORDER_METHOD else f"multiple_dimensions_{type(problem).__name__}_zero_order.pkl"
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), filename) 

    print(f"Running {len(args)} problems")
    results = []
    with multiprocessing.Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(run_algorithm, args), total=len(args)))

    print(f"They are {len(results)} results")

    dictionnary = {}
    for [dimension, suboptimality, chance_constraint_quantile, x_history] in results:
        dictionnary[dimension] = {
            "suboptimality": suboptimality,
            "chance_constraint_quantile": chance_constraint_quantile,
            "x_history": x_history
        }

    pickle.dump(dictionnary, open(OUTPUT_PATH, 'wb'))

if __name__ == "__main__":
    print("Running without zero order method")
    main(ZERO_ORDER_METHOD=False)
    print("Running with zero order method")
    main(ZERO_ORDER_METHOD=True)