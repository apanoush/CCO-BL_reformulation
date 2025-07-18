import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
from tqdm import tqdm
from optimizer import Optimizer
import multiprocessing
from problem import Problem
import pickle
from cpp_paper_problem import CppPaperProblem1
from algorithm import Algorithm


def run_algorithm(args):

    problem, ZERO_ORDER_METHOD = args
    algorithm = Algorithm(problem, verbose=False, empirical_coverage=False, plots=False, zero_order_method=ZERO_ORDER_METHOD)
    opt = Optimizer(algorithm=algorithm)
    opt.run()
    empirical_coverage = algorithm.oracle.empirical_coverage(opt.result)
    f_value = problem.f_function(opt.result)

    return [opt.result, f_value, empirical_coverage]

def main(ZERO_ORDER_METHOD):

    # problem = Problem(
    #     initial_x=0.6, N=5, T=150, I=200, max_iters=500, lambda_=1e-5, GA_steps=1500, theta_G=0.05, theta_H=0.1,
    #     learning_rate_SGLD=2e-2, learning_rate_GD=3e-4, learning_rate_GA=2e1, delta=1e-2, mu=1e-1, K=0.3
    # )
    # problem = Problem(
    #     initial_x=0, N=5, T=300, I=200, max_iters=1000, lambda_=1e-5, GA_steps=1500, theta_G=1-0.95, theta_H=0.1,
    #     learning_rate_SGLD=1e-2, learning_rate_GD=2e-4, learning_rate_GA=2e1, delta=1e-2, mu=1e-2, K=2e-4
    # )
    problem = CppPaperProblem1(
       initial_x=-5, N=10, T=500, I=200, max_iters=150, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
      learning_rate_SGLD=1e-1, learning_rate_GD=1.4e-2, learning_rate_GA=0.1, delta=1e-1, mu=2e0, K=1.4e-2
    )
    filename = f"results_{type(problem).__name__}.pkl" if not ZERO_ORDER_METHOD else f"results_{type(problem).__name__}_zero_order.pkl"
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), filename) 


    num_runs = 100
    results = []
    num_processes = multiprocessing.cpu_count()
    print(f"Running {num_runs} runs with {num_processes} processes")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(run_algorithm, [(problem, ZERO_ORDER_METHOD)] *num_runs), total=num_runs))

    results = np.array(results)

    dictionnary = {}
    dictionnary["x"] = results[:,0]
    dictionnary["f_value"] = results[:,1]
    dictionnary["empirical_coverage"] = results[:,2]

    pickle.dump(dictionnary, open(OUTPUT_PATH, 'wb'))

if __name__ == "__main__":
    main(ZERO_ORDER_METHOD=False)
    main(ZERO_ORDER_METHOD=True)