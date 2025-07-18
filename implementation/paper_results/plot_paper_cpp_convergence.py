import sys
sys.path.insert(0,'.')
import os
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
from tqdm import tqdm
from cycler import cycler
from problem import Problem
from taco_paper_problem import TacoPaperProblem1, TacoPaperProblem2, TacoPaperProblem1_2, TacoPaperProblem1_3
from cpp_paper_problem import CppPaperProblem1, CppPaperProblem2, CppPaperProblem1_0, CppPaperProblemRAsum
from oracle import Oracle
from tqdm import tqdm
from paper_results.compute_empirical_coverage import compute_empirical_coverage
import json

problem = [CppPaperProblem1_0(
       initial_x=[3,3], N=10, T=500, I=500, max_iters=100, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
      learning_rate_SGLD=1e-1, learning_rate_GD=3e-2, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-2, update_clipping=None),
      CppPaperProblemRAsum(
       initial_x=[0,0, 0], N=10, T=500, I=500, max_iters=2000, lambda_=0.1, GA_steps=300, theta_H=0.9,
      learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-4, update_clipping=5
    )
    ][-1]
#TITLE = f"{problem} results from 100 runs"
INPUT_PATH = os.path.join(os.path.dirname(__file__),f"convergence_{type(problem).__name__}.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), f"convergence_plot_{type(problem).__name__}.pdf")

#XLIM = (0, 70)
if isinstance(problem, CppPaperProblem1_0):
    DIRECTORY = "paper_results/cpp/CppProblem1_0"
    XLIM = (0, 150)
elif isinstance(problem, CppPaperProblemRAsum):
    DIRECTORY = "paper_results/cpp/CppProblemRA"
    XLIM = (0, 200)


def import_json(path):
    print(path)
    return json.load(open(path, "r"))


def compute_f_history(problem, x_history):
    """Compute the f_history of the algorithm"""
    f_history = []
    for x in tqdm(x_history, desc="Computing f_history"):
        f_history.append(problem.f_function(x))
    return np.array(f_history)

def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results
    return [pickle.loads(item) for item in results]

if isinstance(problem, CppPaperProblem1_0) or isinstance(problem, CppPaperProblemRAsum):
    files = os.listdir(DIRECTORY)
    files = [f for f in files if f.endswith(".json")]
    print(f"Found {len(files)} files in {DIRECTORY}")
    cpp_results = [(import_json(os.path.join(DIRECTORY, f)), f.split("_")[0]) for f in files]
    cpp_fs = [(compute_f_history(problem, res), title) for res, title in cpp_results]
    #cpp_empirical_coverages = [(compute_empirical_coverage(problem, res), title) for res, title in cpp_results]
    cpp_empirical_coverages = pickle.load(open("paper_results/empirical_coverage_CppPaperProblem_ressource_allocation.pkl", "rb"))



def plot_suboptimality(results, problem, save_path=None, figsize=(9, 3)):
    """(f(x_k)−f^*)/|f^*|
    """
    res, res_zero_order = results

    # assert problem.empirical_coverage, "Suboptimality plot is only available when empirical coverage is True"
    # assert problem.optimal_value is not None, "Optimal value is not available"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    f_history = compute_f_history(problem, res["x_history"][:XLIM[-1]])
    f_history_zero_order = compute_f_history(problem, res_zero_order["x_history"][:XLIM[-1]])
    iterations = np.arange(len(f_history))
    ax1.plot(iterations, f_history, label="Gradient based")
    ax1.plot(iterations, f_history_zero_order, label="Zeroth order")

    for cpp_f, title in cpp_fs:
        # if res not the same shape as supobtimality, repeat the last value
        if len(cpp_f) != iterations.shape[0]:
            cpp_f = np.concatenate((cpp_f, [cpp_f[-1]] * (iterations.shape[0] - len(cpp_f))))
        ax1.plot(iterations, cpp_f, label=title)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Objective function")
    ax1.set_title("Objective function against iterations")
    ax1.set_xlim(*XLIM)
    #ax1.set_yscale("log")
    ax1.grid(True)

    empirical_coverage_suboptimality = res["empirical_coverage_history"][:XLIM[-1]]
    empirical_coverage_suboptimality_zero_order = res_zero_order["empirical_coverage_history"][:XLIM[-1]]


    ax2.axhline(y=(1-problem.theta_G), color='black', linestyle='--', label="$1-\delta$", alpha=0.5)

    ax2.plot(iterations[:XLIM[-1]], empirical_coverage_suboptimality)
    ax2.plot(iterations[:XLIM[-1]], empirical_coverage_suboptimality_zero_order)

    for cpp_empirical_coverage, title in cpp_empirical_coverages:
        # if res not the same shape as supobtimality, repeat the last value
        if len(cpp_empirical_coverage) != len(empirical_coverage_suboptimality):
            cpp_empirical_coverage = np.concatenate((cpp_empirical_coverage, [cpp_empirical_coverage[-1]] * (len(empirical_coverage_suboptimality) - len(cpp_empirical_coverage))))
        ax2.plot(iterations, cpp_empirical_coverage)

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Empirical coverage")
    ax2.set_title("Empirical coverage against iterations")
    #ax2.set_yscale("log")
    ax2.set_xlim(*XLIM)
    ax2.grid(True)

    fig.legend(loc="lower center", ncol=4)
    fig.tight_layout(rect = [0, 0.08, 1, 1])

    if save_path:
        try:
            plt.savefig(save_path)
        except:
            print(f"Could not save the plot to {save_path}")

    plt.show()

def main():
    results = import_results(INPUT_PATH)

    plot_suboptimality(results, problem, save_path=OUTPUT_PATH)#, save_path="results_plot.png")

if __name__ == "__main__":
    main()