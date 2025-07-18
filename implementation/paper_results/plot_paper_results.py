import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from tqdm import tqdm
from cycler import cycler

NUM_PLOTS = 2

problem = ["Problem", "CppPaperProblem1"][-1]
PLOT_WIDTH = {
    "Problem": 5e-2,
    "CppPaperProblem1": 1/8,
}[problem]
#TITLE = f"{problem} results from 100 runs"
INPUT_PATH = os.path.join(os.path.dirname(__file__),f"results_{problem}.pkl")
INPUT_PATH_ZERO_ORDER = os.path.join(os.path.dirname(__file__),f"results_{problem}_zero_order.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), f"results_plot_{problem}.pdf")

def get_xlim(data, min=None, max=None):
    """returns range of 1 around mean of data"""
    mean = np.mean(data)
    width = PLOT_WIDTH
    # making sure the range doesnt exceed min
    if min and mean - width < min:
        mean = min + width
    # making sure the range doesnt exceed max
    if max and mean + width > max:
        mean = max - width
    return (mean - width, mean + width)

def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_results(results, results_zero_order, save_path=None):
    """plots each of the 3 array in a suplot histogram"""

    #plt.rc("text", usetex=True)
    fig, axs = plt.subplots(1, NUM_PLOTS, figsize=(10/3 * NUM_PLOTS, 2))
    #fig.suptitle(title)

    i = -1
    if NUM_PLOTS == 3:
        i = 0
        # Plot x
        axs[i].hist(results["x"], bins=30, alpha=0.6, density=False)
        axs[i].hist(results_zero_order["x"], bins=30, alpha=0.6, density=False)
        axs[i].set_title("$x^*$")
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(get_xlim(results["x"]))
    
    # Plot f_value
    axs[i+1].hist(results["f_value"], bins=30, alpha=0.7, density=False,  label="Gradient based")
    axs[i+1].hist(results_zero_order["f_value"], bins=30, alpha=0.7, density=False, label="Zero order")
    axs[i+1].set_title('$f(x^*)$')
    axs[i+1].set_xlabel('Value')
    axs[i+1].set_ylabel('Frequency')
    axs[i+1].set_xlim(get_xlim(results["f_value"]))

    # Plot empirical_coverage
    axs[i+2].hist(results["empirical_coverage"], bins=30, alpha=0.7, density=False)
    axs[i+2].hist(results_zero_order["empirical_coverage"], bins=30, alpha=0.7, density=False)
    axs[i+2].set_title('Empirical Coverage')
    axs[i+2].set_xlabel('Value')
    axs[i+2].set_ylabel('Frequency')
    axs[i+2].set_xlim(get_xlim(results["empirical_coverage"], max=1))

    # use bbox_to_anchor to place the legend below plot
    fig.legend(loc="lower center", ncol=2)
    #fig.subplots_adjust(bottom=0.15)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    results = import_results(INPUT_PATH)
    results_zero_order = import_results(INPUT_PATH_ZERO_ORDER)

    plot_results(results, results_zero_order, save_path=OUTPUT_PATH)#, save_path="results_plot.png")

if __name__ == "__main__":
    main()