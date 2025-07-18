import pickle
import sys
sys.path.insert(0, ".")
import numpy as np
from oracle import Oracle
from taco_paper_problem import TacoPaperProblem2
INITIAL_X = {d: [0.1] * d for d in [2, 10, 50, 200]}

dict_problems = {d: TacoPaperProblem2(
        initial_x=initial_x, N=10, T=300, I=300, GA_steps=500, max_iters=1200, theta_G=1-0.8, theta_H=0.9, lambda_=0, learning_rate_SGLD=6e-1, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=2e1, delta= 7e-1, K=1e-2
    ) for d, initial_x in INITIAL_X.items()}

ZERO_ORDER_METHOD = False  
INPUT = "paper_results/multiple_dimensions_TacoPaperProblem2.pkl" if not ZERO_ORDER_METHOD else "paper_results/multiple_dimensions_TacoPaperProblem2_zero_order.pkl"

def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)

    original_results = results.copy()
    
    # create a dictionary for which each dimension is associated with a tuple containing the suboptimality and the chance_constraint_quantile
    for dim, res in results.items():
        dim_array = []
        for i in range(len(res["suboptimality"])):
            dim_array.append([res["suboptimality"][i], res["chance_constraint_quantile"][i]])
        results[dim] = np.array(dim_array)
        #print("okok", results[dim])

    return results, original_results



def get_best_points(results, x_history, dim):

    #print(results)

    # we filter out the points for which x[2] > 0
    results = results[results[:, 1] > 0]
    # we now sort the results by descending order of x[1]
    results = results[np.argsort(results[:, 1])]
    # we now keep only the first 10 points
    results = results[:3]

    
    
    ec = []
    for i in range(len(results)):
        oracle = Oracle(dict_problems[dim])
        # we compute ec
        ec.append(oracle.empirical_coverage(x_history[i], N=200))

    print("ec", ec)

    # we hstack the results
    results = np.hstack((results, np.array(ec).reshape(-1, 1)))


    return results[0]

def main():
    results, original_results = import_results(INPUT)
    final_res = {}
    for dim, res in results.items():
        point = get_best_points(res, original_results[dim]["x_history"], dim)
        final_res[dim] = point
    
    print(final_res)
    
if __name__ == "__main__":
    main()