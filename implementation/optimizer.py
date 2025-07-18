import numpy as np
import sys
sys.path.append('.')
from problem import Problem
from taco_paper_problem import TacoPaperProblem1, TacoPaperProblem2, TacoPaperProblem1_2, TacoPaperProblem1_3
from cpp_paper_problem import CppPaperProblem1, CppPaperProblem2, CppPaperProblem1_0, CppPaperProblemRAsum
from oracle import Oracle
from algorithm import Algorithm
from plots import *
from cepdd_paper_problem import CepddPaperProblem1, CepddPaperProblem2
from problem_instances import problem_instances

class Optimizer:
    """main class, containing the problem, the oracle, and the algorithm"""

    def __init__(self, algorithm: Algorithm):
        self.algorithm = algorithm
        self.result = None
    
    def run(self):

        self.result = self.algorithm.run()

if __name__ == "__main__":

    problem_name = ["CppPaperProblem_RA_max", "CppPaperProblem2_3"][-1]
    problem = problem_instances[problem_name]
 
    algorithm = Algorithm(problem, verbose=True, empirical_coverage=False, plots=False, zero_order_method=False)

    optimizer = Optimizer(algorithm=algorithm)
    optimizer.run()

    # get empirical coverage of solution
    empirical_coverage = optimizer.oracle.empirical_coverage(optimizer.result)
    f_value = problem.f_function(optimizer.result)
    tqdm.write(f"Empirical coverage: {empirical_coverage}")
    tqdm.write(f"function value: {f_value}")

    optimizer.oracle.save_results(optimizer.result, f_value, empirical_coverage, type(problem).__name__)

    if isinstance(problem, CppPaperProblem2):
        new_x_history = np.zeros(len(optimizer.algorithm.objective_function_history))
        for i, bloc in enumerate(optimizer.algorithm.x_history):
            new_x_history[i] = bloc[0, 0]
        
        optimizer.algorithm.x_history = new_x_history

        print(optimizer.algorithm.x_history.shape, len(optimizer.algorithm.objective_function_history))
        plot_results_1D(optimizer.algorithm, problem)
    elif problem.dimension == 1:
        plot_results_1D(optimizer.algorithm, problem)
    elif problem.dimension == 2:    
        plot_results_2D(optimizer.algorithm, problem)

    if optimizer.algorithm.empirical_coverage and optimizer.problem.optimal_solution is not None:
        pass
        #plot_suboptimality(optimizer.algorithm, savepath="implementation/paper_results/Convergence of the algorithm.png")
    
