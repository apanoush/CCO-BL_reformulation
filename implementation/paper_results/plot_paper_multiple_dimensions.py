import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from tqdm import tqdm
import sys
sys.path.insert(0, ".")
from problem import Problem
from taco_paper_problem import TacoPaperProblem2
from oracle import Oracle

problem = "TacoPaperProblem2"
TITLE = f"{problem} results from 100 runs"
INPUT_PATH = os.path.join(os.path.dirname(__file__),f"multiple_dimensions_{problem}.pkl")
INPUT_PATH_ZERO_ORDER = os.path.join(os.path.dirname(__file__),f"multiple_dimensions_{problem}_zero_order.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), f"multiple_dimensions_plot_{problem}.pdf")

X_RANGES = {
    2: (0, 1000),
    10: (0, 400),
    50: (0, 400),
    200: (0, 400)
}

def preprocess_taco_results(path):
    problem = TacoPaperProblem2(
        initial_x=[2, 1], N=10, T=50, I=400, GA_steps=500, max_iters=700, theta_G=1-0.95, theta_H=0.9, lambda_=0, learning_rate_SGLD=5e0, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=1, delta= 5, K=1e-2)

    res = pickle.load(open(path, 'rb'))
    res = res[2]
    res["suboptimality"] = np.array([abs(problem.f_function(x) - problem.optimal_value) / abs(problem.optimal_value) for x in res["x_history"]])
    #res["empircal_coverage"] = oracle.empirical_coverage(res["x_history"], 100)
    return res



def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_results(results, resuts_zero_order, save_path=None):
    """plots each of the 3 array in a suplot histogram"""

    res2 = preprocess_taco_results("paper_results/taco_res/res.pkl")

    fig, axs = plt.subplots(2, 2, figsize=(11, 5))
    for ax, dim in zip(axs.flatten(), results.keys()): 
        res = results[dim]
        iterations = np.arange(len(res["suboptimality"]))
        ax.plot(iterations, res["suboptimality"], alpha=1, label="Gradient based" if dim == 2 else "")
        ax.plot(iterations, resuts_zero_order[dim]["suboptimality"], alpha=0.7, label="Zero order" if dim == 2 else "", color="red")
        #if dim == 2: ax.plot(iterations, res2["suboptimality"][0:700], alpha=0.5, label="Empirical coverage")
        #test = (res["chance_constraint_quantile"])
        # find all points where the chance constraint is changing sign
        index = np.where(np.abs(np.diff(np.sign((res["chance_constraint_quantile"]))) == 2))[0] 
        index = np.append(index, len(res["chance_constraint_quantile"]) - 1) # add the last index
        last_ind = 0
        red = False
        for i, ind in enumerate(index):
            if not red: 
                ax.axvspan(last_ind, ind, color='green', alpha=0.2, label="Feasible solution (Gradient based)" if dim == 2 and i == 0 else "")
                red = True
            else: 
                ax.axvspan(last_ind, ind, color='red', alpha=0.2, label="Infeasible solution (Gradient based)" if dim == 2 and i == 1 else "")
                red = False
            last_ind = ind
        ax.set_title(f"$d={dim}$")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Sub-optimality')
        ax.set_yscale('log')

        ax.set_xlim(X_RANGES[dim])
        
    fig.legend(loc="lower center", ncol=len(axs.flatten()))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    results = import_results(INPUT_PATH)
    results_zero_order = import_results(INPUT_PATH_ZERO_ORDER)

    plot_results(results, resuts_zero_order=results_zero_order, save_path=OUTPUT_PATH)#, save_path="results_plot.png")

if __name__ == "__main__":
    main()