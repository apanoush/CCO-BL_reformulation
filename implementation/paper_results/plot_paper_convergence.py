import sys
sys.path.insert(0,'.')
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
from tqdm import tqdm
from oracle import Oracle
from paper_results import compute_empirical_coverage
import json
from problem_instances import problem_instances


problem_name = ["Problem", "TacoPaperProblem1", "CppPaperProblem1", "CppPaperProblem1_0", "CppPaperProblemRAsum", "CppPaperProblemRAmax", "TacoPaperProblem1_2", "TacoPaperProblem1_3"][-4]
problem = problem_instances[problem_name]

suboptimality = False if "Taco" in problem_name or "Cpp" in problem_name else True
ec_pickle = True# if "Taco" in problem_name or "Cpp" in problem_name or problem_name == "Problem" else False

# separating our, TACO and CPP in two distinct plots
SPLIT = False

INPUT_PATH = os.path.join(os.path.dirname(__file__),f"convergence_{type(problem).__name__}.pkl")
OUTPUT_PATH = f"paper_results/plots/convergence_plot_{type(problem).__name__}.pdf"
OUTPUT_PATH_LATEX = f"paper_results/tables/table_{type(problem).__name__}.tex"
EC_PICKLE_PATH = os.path.join(os.path.dirname(__file__), f"empirical_coverage_{problem_name}.pkl")
EC_PICKLE_PATH_TACO = os.path.join(os.path.dirname(__file__), f"empirical_coverage_{problem_name}_bundle.pkl")

XLIM = {
    "TacoPaperProblem": (0, 100),
    "TacoPaperProblem1": (0, 100)
}.get(problem_name, None)

# YLIMS for sub_optimality
YLIM1 = {
    "Problem": (-0.02, 1/2)
}.get(problem_name, None)

def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results
    return [pickle.loads(item) for item in results]

def import_json(path):
    return json.load(open(path, "r"))

def import_ec(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results

def length_freeze(res, length):
    if len(res) != length:
        res = np.concatenate((res, [res[-1]] * (length - len(res))))
    return res

def get_results(results, suboptimality):
    res, res_zero_order = results

    if suboptimality:
        suboptimality = Oracle.suboptimality_f(problem, res["x_history"])
        suboptimality_zero_order =  Oracle.suboptimality_f(problem, res_zero_order["x_history"])
    else: 
        # suboptimality = res["f_history"]
        # suboptimality_zero_order = res_zero_order["f_history"]
        suboptimality = compute_f_history(problem, res["x_history"])
        suboptimality_zero_order = compute_f_history(problem, res_zero_order["x_history"])
    if XLIM:
        suboptimality = suboptimality[:XLIM[-1]]
        suboptimality_zero_order = suboptimality_zero_order[:XLIM[-1]]
    return res, res_zero_order, suboptimality, suboptimality_zero_order

def get_cpp_results():
    if "RA" not in problem_name:
        dir_ = f"paper_results/cpp/{problem_name}"        
        files = os.listdir(dir_)
        print(["SAA" in f for f in files])
        files = [f for f in files if f.endswith(".json") and ("SAA" in f or "CPP" in f)]
    else:
        dir_ = f"paper_results/cpp/CppPaperProblemRA"        
        files = os.listdir(dir_)
        if "max" in problem_name:
            max_or_union = "Max"
        elif "sum" in problem_name:
            max_or_union = "Union"
        else:
            raise Exception
        files = [f for f in files if max_or_union in f and f.endswith(".json") and ("SAA" in f or "CPP" in f)]

    print(f"Found {len(files)} files in {dir_}")
    cpp_results = [(import_json(os.path.join(dir_, f)), f.split("_")[0]) for f in files]
    if suboptimality:
        cpp_suboptimality = [(Oracle.suboptimality_f(problem, list(res.values())[0]), title) for res, title in cpp_results]
    else:
        cpp_suboptimality = []
        for res, title in cpp_results:
            values = []
            for value in list(res.values())[0]:
                values.append(problem.f_function(value))
            cpp_suboptimality.append((values, title))
    
    #if not ec_pickle:
    cpp_empirical_coverage = [(compute_empirical_coverage.compute_empirical_coverage(problem, list(res.values())[0]), title) for res, title in cpp_results]
    #else:
    #    cpp_empirical_coverage = import_ec(EC_PICKLE_PATH)
    return cpp_suboptimality, cpp_empirical_coverage

def get_taco_results():
    dir_ = f"paper_results/taco_res/"
    file = f"output_{problem_name}.json"
    taco_result = import_json(os.path.join(dir_, file))
    x_history = trim_repeated_end(taco_result["x_history"])
    if suboptimality:
        taco_suboptimality = Oracle.suboptimality_f(problem, x_history)
    else: taco_suboptimality = compute_f_history(problem, x_history)
    if not ec_pickle: taco_empirical_coverage = compute_empirical_coverage.compute_empirical_coverage(problem, x_history)
    else:
        taco_empirical_coverage = import_ec(EC_PICKLE_PATH_TACO)

    if XLIM:
        taco_suboptimality = taco_suboptimality[:XLIM[-1]]
        taco_empirical_coverage = taco_empirical_coverage[:XLIM[-1]]
    return taco_suboptimality, taco_empirical_coverage

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

def compute_f_history(problem, x_history):
    """Compute the f_history of the algorithm"""
    f_history = []
    for x in tqdm(x_history, desc="Computing f_history"):
        f_history.append(problem.f_function(x))
    return f_history

def plot_suboptimality(results, results_taco, results_cpp, problem, save_path=None, figsize=(9, 3), split=None, y_lims = None):
    """(f(x_k)−f^*)/|f^*|
    """

    res, res_zero_order, res_suboptimality, res_suboptimality_zero_order = results
    taco_suboptimality, taco_empirical_coverage = results_taco
    if results_cpp:
        cpp_suboptimality, cpp_empirical_coverage = results_cpp

    # assert problem.empirical_coverage, "Suboptimality plot is only available when empirical coverage is True"
    # assert problem.optimal_value is not None, "Optimal value is not available"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    length = len(res_suboptimality) if split == 1 or not split else max([len(cpp_res) for cpp_res, _ in cpp_suboptimality])
    iterations = np.arange(length)

    if XLIM:
        iterations = iterations[:XLIM[-1]]
    taco_suboptimality = taco_suboptimality[:len(iterations)]
    taco_empirical_coverage = taco_empirical_coverage[:len(iterations)]

    if split == 1 or not split:
        ax1.plot(iterations, res_suboptimality, label="Algorithm 1")
        ax1.plot(iterations, res_suboptimality_zero_order, label="Algorithm 2")

        taco_suboptimality = length_freeze(taco_suboptimality, length)
        ax1.plot(iterations, taco_suboptimality, label="Bundle")

    if (split == 2 or not split) and cpp_suboptimality:
        for cpp_res, title in cpp_suboptimality:
            # if res not the same shape as supobtimality, repeat the last value
            cpp_res = length_freeze(cpp_res, length)

            ax1.plot(iterations, cpp_res, label=title)        

    ax1.set_xlabel("Iterations")
    ylabel = "Sub-optimality" if suboptimality else "Objective function"
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{ylabel} per iteration")
    #ax1.set_yscale("log")
    # if XLIM:
    #     ax1.set_xlim(XLIM[-1])
    ax1.grid(True)

    ax2.axhline(y=(1-problem.theta_G), color='black', linestyle='--', label="$1-\delta$", alpha=0.5)

    if ec_pickle: #and not "Taco" in problem_name:
        print("Loading already computed empirical coverages")
        res_empirical_coverage, res_empirical_coverage_zero_order = import_ec(EC_PICKLE_PATH)
    else:
        res_empirical_coverage = res["empirical_coverage_history"]
        res_empirical_coverage_zero_order = res_zero_order["empirical_coverage_history"]

    if split == 1 or not split:
        res_empirical_coverage = res_empirical_coverage[:len(iterations)]
        res_empirical_coverage_zero_order = res_empirical_coverage_zero_order[:len(iterations)]

        ax2.plot(iterations, res_empirical_coverage)#[:XLIM[-1]])
        ax2.plot(iterations, res_empirical_coverage_zero_order)

        taco_empirical_coverage = length_freeze(taco_empirical_coverage, length)
        ax2.plot(iterations, taco_empirical_coverage)

    if (split == 2 or not split) and cpp_empirical_coverage:
        for empirical_coverage_, title in cpp_empirical_coverage:
            # if res not the same shape as subobtimality, repeat the last value
            empirical_coverage_ = length_freeze(empirical_coverage_, length)
            ax2.plot(iterations, empirical_coverage_)

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Empirical coverage")
    ax2.set_title("Empirical coverage per iteration")
    #ax2.set_yscale("log")
    ax2.grid(True)

    # useful to keep both yaxis max same if split 

    y_lim1 = ax1.get_ylim()
    y_lim2 = ax2.get_ylim()

    # y_lim1 = ax1.get_ylim()
    # y_lim2 = ax2.get_ylim()  

    if YLIM1: y_lim1 = YLIM1          

    ax1.set_ylim(y_lim1)
    ax2.set_ylim(y_lim2)


    ncol = {
        None: 4,
        1: 4,
        2: 5
    }[split]

    if "RA" in problem_name: ncol = 6

    fig.legend(loc="lower center", ncol=ncol)

    if "Taco" or "RA" in problem_name:
        padding = 0.08  
    else:
        padding = 0.15

    fig.tight_layout(rect = [0, padding, 1, 1])

    if split == 1:
        save_path = save_path.replace(".pdf", "_1.pdf")
    elif split == 2:
        save_path = save_path.replace(".pdf", "_2.pdf")

    if save_path:
        try:
            plt.savefig(save_path)
        except:
            print(f"Could not save the plot to {save_path}")

    plt.show()

    return res_suboptimality, res_empirical_coverage, res_suboptimality_zero_order, res_empirical_coverage_zero_order, taco_suboptimality, taco_empirical_coverage, cpp_suboptimality, cpp_empirical_coverage
    
def prepare_table_dict(results, empirical_coverage, last_n=10):
    """
    Takes 2 np arrays as inputs and outputs the mean of their last 10 elements
    
    Args:
        results: numpy array of results
        empirical_coverage: numpy array of empirical coverage values
    
    Returns:
        dict: Dictionary with mean of last 10 elements for each input array
    """
    results_mean = np.mean(results[-last_n:])
    coverage_mean = np.mean(empirical_coverage[-last_n:])
    
    return {
        'final_result': results_mean,
        'empirical_coverage': coverage_mean
    }


def generate_latex_table(results, filename, suboptimality, round_digits=4, coverage_percentage=False):
    """
    Generates a LaTeX table from algorithm results.
    
    Args:
        results: Dictionary of {
                    'Algorithm Name': {
                        'final_result': float, 
                        'empirical_coverage': float
                    }
                }
        filename: Output .tex filename
        round_digits: Rounding precision for numbers
        coverage_percentage: Format coverage as percentage (0-100 scale)
    """
    # Header configuration
    #label = "Sub-optimality" if suboptimality else "Objective function"
    if suboptimality:
        latex_header = r"""\begin{table}[ht]
    \centering
    \begin{tabular}{lcc}
    \toprule
    \textbf{Algorithm} & \textbf{Sub-optimality} & \textbf{Empirical Coverage} \\
    \midrule
    """
    else:
        latex_header = r"""\begin{table}[ht]
    \centering
    \begin{tabular}{lcc}
    \toprule
    \textbf{Algorithm} & \textbf{Objective function} & \textbf{Empirical coverage} \\
    \midrule
    """

    latex_footer = r"""\bottomrule
    \end{tabular}
    \caption{Algorithm performance: average of the last iterations}
    \end{table}"""

    # Process each algorithm
    rows = []
    for algo, values in results.items():
        # Format numerical values
        final_val = round(values['final_result'], round_digits)
        
        # Format coverage (percentage or decimal)
        cov_val = values['empirical_coverage']
        if coverage_percentage:
            cov_val = round(cov_val * 100, round_digits)
            cov_str = f"{cov_val:.{round_digits}f}\\%"
        else:
            cov_str = f"{cov_val:.{round_digits}f}"
        
        # Escape special LaTeX characters in algorithm names
        algo_escaped = algo.replace('_', '\\_')
        
        rows.append(f"{algo_escaped} & ${final_val}$ & ${cov_str}$ \\\\")

    # Combine all components
    latex_content = latex_header + "\n".join(rows) + "\n" + latex_footer
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(latex_content)


def main():
    results = import_results(INPUT_PATH)
    results = get_results(results, suboptimality)
    taco_results = get_taco_results()
    SPLIT = False
    if "Taco" not in problem_name:
        cpp_results = get_cpp_results()
    else:
        cpp_results = (None, None)
        SPLIT = False

    res_suboptimality, res_empirical_coverage, res_suboptimality_zero_order, res_empirical_coverage_zero_order, taco_suboptimality, taco_empirical_coverage, cpp_suboptimality, cpp_empirical_coverage = plot_suboptimality(results, taco_results, cpp_results, problem, save_path=OUTPUT_PATH, split=None, y_lims=None)

    results_table = {
        "Algorithm 1": prepare_table_dict(res_suboptimality, res_empirical_coverage),
        "Algorithm 2": prepare_table_dict(res_suboptimality_zero_order, res_empirical_coverage_zero_order),
        "Bundle": prepare_table_dict(taco_suboptimality, taco_empirical_coverage)
    }
    if cpp_suboptimality:
        results_table.update({cpp_suboptimality[i][-1]: prepare_table_dict(cpp_suboptimality[i][0], cpp_empirical_coverage[i][0], last_n=1) for i in range(len(cpp_suboptimality))})

    generate_latex_table(results_table, OUTPUT_PATH_LATEX, suboptimality)

    print(f"Results saved in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()