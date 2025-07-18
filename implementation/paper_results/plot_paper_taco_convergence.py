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
from cpp_paper_problem import CppPaperProblem1, CppPaperProblem2, CppPaperProblem1_0
from oracle import Oracle
from tqdm import tqdm

problem = [TacoPaperProblem1(initial_x=[0, 0], N=10, T=2000, I=2000, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e0, learning_rate_GA=7e4, learning_rate_GD=2e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2),
            TacoPaperProblem1_2(
       initial_x=[1, -1], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2
    ),TacoPaperProblem1_3(
       initial_x=[-2, 3], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=1.5e-1, update_clipping=1/2)
    ][0]
#TITLE = f"{problem} results from 100 runs"
INPUT_PATH = os.path.join(os.path.dirname(__file__),f"convergence_{type(problem).__name__}.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), f"convergence_plot_{type(problem).__name__}.pdf")

#XLIM = (0, 70)
XLIM = (0, 100)
INPUT_OTHER_ALGO = "paper_results/taco_res/output_ToyProblem.pkl"
taco_path = pickle.load(open(INPUT_OTHER_ALGO, "rb"))
INPUT_OTHER_ALGO_EC = os.path.join(os.path.dirname(__file__), f"empirical_coverage_{type(problem).__name__}.pkl")



def compute_f_history(problem, x_history):
    """Compute the f_history of the algorithm"""
    f_history = []
    for x in tqdm(x_history, desc="Computing f_history"):
        f_history.append(problem.f_function(x))
    return f_history

def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results
    return [pickle.loads(item) for item in results]

def plot_suboptimality(results, problem, save_path=None, figsize=(9, 3)):
    """(f(x_k)−f^*)/|f^*|
    """
    res, res_zero_order = results

    # assert problem.empirical_coverage, "Suboptimality plot is only available when empirical coverage is True"
    # assert problem.optimal_value is not None, "Optimal value is not available"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    f_taco = compute_f_history(problem, taco_path["x_history"][:XLIM[-1], :2])
    f_history = compute_f_history(problem, res["x_history"][:XLIM[-1]])
    f_history_zero_order = compute_f_history(problem, res_zero_order["x_history"][:XLIM[-1]])
    iterations = np.arange(len(f_history))
    ax1.plot(iterations, f_history, label="Gradient based")
    ax1.plot(iterations, f_history_zero_order, label="Zeroth order")
    ax1.plot(iterations, f_taco, label="Bundle algorithm")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Objective function")
    ax1.set_title("Objective function against iterations")
    ax1.set_xlim(*XLIM)
    #ax1.set_yscale("log")
    ax1.grid(True)

    empirical_coverage_suboptimality = res["empirical_coverage_history"][:XLIM[-1]]
    empirical_coverage_suboptimality_zero_order = res_zero_order["empirical_coverage_history"][:XLIM[-1]]

    #taco_EC = compute_empirical_coverage(problem, taco_path["x_history"][:XLIM[-1], :2])
    taco_EC = import_results(INPUT_OTHER_ALGO_EC)[:XLIM[-1]]
    # empirical_coverage_suboptimality = compute_empirical_coverage(problem, empirical_coverage_suboptimality)
    # empirical_coverage_suboptimality_zero_order = compute_empirical_coverage(problem, empirical_coverage_suboptimality_zero_order)


    ax2.axhline(y=(1-problem.theta_G), color='black', linestyle='--', label="$1-\delta$", alpha=0.5)

    ax2.plot(iterations[:XLIM[-1]], empirical_coverage_suboptimality)
    ax2.plot(iterations[:XLIM[-1]], empirical_coverage_suboptimality_zero_order)
    ax2.plot(iterations[:XLIM[-1]], taco_EC)
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