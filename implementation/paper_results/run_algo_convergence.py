import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
from tqdm import tqdm
from optimizer import Optimizer
import multiprocessing
from taco_paper_problem import TacoPaperProblem1, TacoPaperProblem2, TacoPaperProblem1_2, TacoPaperProblem1_3
from cpp_paper_problem import CppPaperProblem1, CppPaperProblem2, CppPaperProblem1_0, CppPaperProblemRAsum
from problem import Problem
import pickle as pickle
import json
from algorithm import Algorithm
from copy import deepcopy
from problem_instances import problem_instances

ONLY_RUN_ZERO_ORDER = False

def run_algorithm(args):

    problem, ZERO_ORDER_METHOD = args
    algorithm = Algorithm(problem, verbose= ZERO_ORDER_METHOD, empirical_coverage=False, plots=False, zero_order_method=ZERO_ORDER_METHOD)
    opt = Optimizer(algorithm)
    opt.run()
    # empirical_coverage = opt.oracle.empirical_coverage(opt.result)
    # f_value = problem.f_function(opt.result)

    return {"x_history": opt.algorithm.x_history, "f_history": opt.algorithm.objective_function_history, "chance_constraint_history": opt.algorithm.chance_constraint_quantile_history, "empirical_coverage_history": opt.algorithm.empirical_coverage_history, "problem": type(problem).__name__, "ZERO_ORDER_METHOD": ZERO_ORDER_METHOD}

def main():

    # problem = Problem(
    #     initial_x=0, N=5, T=300, I=200, max_iters=1000, lambda_=1e-5, GA_steps=1500, theta_G=1-0.95, theta_H=0.1,
    #     learning_rate_SGLD=1e-2, learning_rate_GD=2e-4, learning_rate_GA=2e1, delta=1e-2, mu=1e-2, K=2e-4
    # )
    # problem = CppPaperProblem1(
    #    initial_x=-6, N=10, T=500, I=200, max_iters=600, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=1.4e-2, learning_rate_GA=0.1, delta=1e-1, mu=2e0, K=1.4e-2
    # )
    # problem = CppPaperProblem1(
    #    initial_x=-5, N=10, T=500, I=500, max_iters=800, lambda_=0.1, GA_steps=300, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, learning_rate_GA=0.1, delta=5e-2, mu=2e0, K=3e-4, update_clipping=10
    # ) #5e-3
    # problem = TacoPaperProblem1(
    #    initial_x=[0, 0], N=10, T=2000, I=2000, GA_steps=500, max_iters=100, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e0, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=1e-3, delta= 1e-2, K=1e-2, update_clipping=5
    # )
    # problem = TacoPaperProblem1_2(
    #    initial_x=[1, -1], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2
    # )
    # problem = TacoPaperProblem1_3(
    #    initial_x=[-2, 3], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=1.5e-1, update_clipping=1/2
    # )
    # problem = CppPaperProblem1_0(
    #    initial_x=[3,3], N=10, T=500, I=500, max_iters=500, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=3e-2, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-2, update_clipping=None
    # )
    # problem = CppPaperProblem_ressource_allocation(
    #    initial_x=[0,0, 0], N=10, T=500, I=500, max_iters=200, lambda_=0.1, GA_steps=300, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-4, update_clipping=5
    # )

    problem_name = ["Problem", "TacoPaperProblem1_2", "TacoPaperProblem1_3", "CppPaperProblemRAmax", "CppPaperProblemRAsum", "CppPaperProblem1_0"][4]
    problem = problem_instances[problem_name]

    problems = [(deepcopy(problem), False), (deepcopy(problem), True)]

    filename = f"convergence_{type(problem).__name__}.pkl" 
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), filename) 

    if ONLY_RUN_ZERO_ORDER: 
        problems.pop(0)
        try:
            pickle.load(open(OUTPUT_PATH, "rb"))[0]
        except:
            print(f"Didn't find {OUTPUT_PATH}, exiting")
            return 1

    results = []
    with multiprocessing.Pool(processes=2) as pool:
        results = list(tqdm(pool.imap(run_algorithm, problems), total=len(problems)))

    if ONLY_RUN_ZERO_ORDER: results.insert(0, pickle.load(open(OUTPUT_PATH, "rb"))[0])

    print(f"They are {len(results)} results")
    pickle.dump(results, open(OUTPUT_PATH, 'wb'))

if __name__ == "__main__":
    main()