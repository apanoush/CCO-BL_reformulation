import sys
sys.path.insert(0,'.')
import os
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
import json
from tqdm import tqdm
from cycler import cycler
from oracle import Oracle
from tqdm import tqdm
from problem_instances import problem_instances

N = 100#100

# problem = [Problem(initial_x=0, N=5, T=300, I=500, max_iters=1000, lambda_=1e-5, GA_steps=1500, theta_G=1-0.95), CppPaperProblem1(
#             initial_x=-6, N=10, T=500, I=50, max_iters=500, theta_G=1-0.95), 
#             TacoPaperProblem1(initial_x=[0, 0], N=10, T=2000, I=2000, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e0, learning_rate_GA=7e4, learning_rate_GD=2e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2), TacoPaperProblem1_2(
#        initial_x=[1, -1], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2
#     ),TacoPaperProblem1_3(
#        initial_x=[-2, 3], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=1.5e-1, update_clipping=1/2
#     ),
#     CppPaperProblem1_0(
#        initial_x=[3,3], N=10, T=500, I=500, max_iters=100, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
#       learning_rate_SGLD=1e-1, learning_rate_GD=3e-2, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-2, update_clipping=None
#     ),
#     CppPaperProblemRAsum(
#        initial_x=[0,0, 0], N=10, T=500, I=500, max_iters=2000, lambda_=0.1, GA_steps=300, theta_H=0.9,
#       learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-4, update_clipping=5
#     )][-1]
#TITLE = f"{problem} results from 100 runs"
problem_name = ["TacoPaperProblem1_2", "TacoPaperProblem1_3", "CppPaperProblem1", "CppPaperProblem1_0", "Problem", "CppPaperProblemRAmax", "CppPaperProblemRAsum", "TacoPaperProblem1"][-2]
problem = problem_instances[problem_name]

INPUT_PATH = os.path.join(os.path.dirname(__file__),f"convergence_{type(problem).__name__}.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), f"empirical_coverage_{type(problem).__name__}.pkl")

XLIM = (0, 120)

INPUT_TACO = f"paper_results/taco_res/output_{problem_name}.json"
DIRECTORY = "paper_results/cpp/CppProblemRA"
OUTPUT_PATH_TACO = os.path.join(os.path.dirname(__file__), f"empirical_coverage_{type(problem).__name__}_bundle.pkl")

input_file = pickle.load(open(INPUT_PATH, "rb"))
taco_file = json.load(open(INPUT_TACO, "r"))

def trim_repeated_end(arr):
    """Remove repeated elements from the end of an array."""
    if len(arr) <= 1:
        return arr
    
    arr = np.array(arr)
    last_value = arr[-1]
    
    # Find the first index from the end where the value differs from the last value
    for i in range(len(arr) - 2, -1, -1):
        if (arr[i] != last_value).all():
            return arr[:i + 2]  # Keep one instance of the repeated value
    
    # If all elements are the same, return just the first element
    return arr[:1]

def import_json(path):
    print(path)
    return json.load(open(path, "r"))

def compute_empirical_coverage(problem, x_history, N=500):
    """Compute the empirical coverage of the algorithm"""
    oracle = Oracle(problem)
    empirical_coverage = []
    for x in tqdm(x_history, desc="Computing empirical coverage"):
        empirical_coverage.append(oracle.empirical_coverage(x, N=N))
    return empirical_coverage


if __name__ == "__main__":

    #if "Taco" in problem_name:
        # other_algo_x = other_algo_path["x_history"][:XLIM[-1], :2]


    x_history_1 = trim_repeated_end(np.array(input_file[0]["x_history"]))
    x_history_2 = trim_repeated_end(np.array(input_file[1]["x_history"]))
    empirical_coverage = (
        compute_empirical_coverage(problem, x_history_1, N=N),
        compute_empirical_coverage(problem, x_history_2, N=N)
    )

    x_history_taco = trim_repeated_end(np.array(taco_file["x_history"]))
    empirical_coverage_taco = compute_empirical_coverage(problem, x_history_taco, N=N)


    #elif "Cpp" in problem_name:
    # files = os.listdir(DIRECTORY)
    # files = [f for f in files if f.endswith(".json")]
    # print(f"Found {len(files)} files in {DIRECTORY}")
    # cpp_results = [(import_json(os.path.join(DIRECTORY, f)), f.split("_")[0]) for f in files]
    # empirical_coverage = [(compute_empirical_coverage(problem, res), title) for res, title in cpp_results]
    
    pickle.dump(empirical_coverage, open(OUTPUT_PATH, 'wb'))
    pickle.dump(empirical_coverage_taco, open(OUTPUT_PATH_TACO, 'wb'))