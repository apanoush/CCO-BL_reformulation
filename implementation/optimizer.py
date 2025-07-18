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

    # problem = Problem(
    #     initial_x=0.2, N=5, T=300, I=200, max_iters=1000, lambda_=1e-5, GA_steps=1500, theta_G=1-0.95, theta_H=0.1,
    #     learning_rate_SGLD=1e-2, learning_rate_GD=2e-4, learning_rate_GA=2e1, delta=1e-2, mu=1e-2, K=2e-4
    # ) # 8e-6 2e-5
    # problem = TacoPaperProblem1(
    #    initial_x=[0, 0], N=10, T=2000, I=2000, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e0, learning_rate_GA=7e4, learning_rate_GD=2e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2
    # )#0.03368421
    # problem = TacoPaperProblem1(
    #    initial_x=[2,1], N=10, T=400, I=400, GA_steps=500, max_iters=400, theta_G=1-0.8, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e2, learning_rate_GA=7e4, learning_rate_GD=1e-8, mu=1, delta= 5, K=1e-5
    # )#0.03368421
    # problem = TacoPaperProblem2(
    #    initial_x=[3,3], N=10, T=50, I=400, GA_steps=500, max_iters=500, theta_G=1-0.8, theta_H=0.9, lambda_=0, learning_rate_SGLD=5e0, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=1, delta= 5, K=1e-5
    # )
    # problem = TacoPaperProblem2(
    #     initial_x=[0.1]*2, N=10, T=300, I=300, GA_steps=500, max_iters=700, theta_G=1-0.8, theta_H=0.9, lambda_=0, learning_rate_SGLD=6e-1, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=2e1, delta= 7e-1, K=1e-2
    # )
    # problem = CppPaperProblem1(
    #    initial_x=-5, N=10, T=500, I=200, max_iters=1000, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=5e-3, learning_rate_GA=0.1, delta=1e-1, mu=2e0, K=5e-3, update_clipping=10
    # ) #1.4e-2
    # problem = CppPaperProblem2(
    #     initial_x=3*np.ones((3, 2)), N=10, T=int(5e4), I=50, max_iters=1000, lambda_=0.1, GA_steps=200, theta_G=0.05, theta_H=0.2,
    #     learning_rate_SGLD=7e3, learning_rate_GD=1e-5, learning_rate_GA=0.3, delta=2, mu=5, num_robot_steps=2, K=0.5
    # )
    # problem = TacoPaperProblem1_2(
    #    initial_x=[-0.5, -1.5], N=10, T=2000, I=300, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=6e0, learning_rate_GA=1e5, learning_rate_GD=1e-2, mu=1e-3, delta= 1e-2, K=1e-2, update_clipping=3
    # ) #1.5e-1
    # problem = TacoPaperProblem1(
    #    initial_x=[0, 0], N=10, T=2000, I=300, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e0, learning_rate_GA=1e5, learning_rate_GD=1e-2, mu=1e-3, delta= 1e-2, K=1e-2, update_clipping=3
    # ) 
    # problem = TacoPaperProblem1_3(
    #    initial_x=[-2, 3], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=1.5e-1, update_clipping=1/2
    # )


    # problem = CepddPaperProblem1(
    #    initial_x=np.ones(8)*5, N=10, T=500, I=200, max_iters=150, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=1.4e-1, learning_rate_GA=0.1, delta=1e-1, mu=2e0, K=1.4e-2
    # )
    # problem = CepddPaperProblem2(
    #    initial_x=np.ones(9)*5, N=10, T=500, I=200, max_iters=150, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=1.4e-1, learning_rate_GA=0.1, delta=1e-1, mu=2e0, K=1.4e-2
    # )
    # problem = CppPaperProblem1(
    #    initial_x=-6.35, N=10, T=500, I=500, max_iters=2000, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=1e-4, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=2e-3, update_clipping=None
    # ) #5e-3
    # problem = CppPaperProblem1_0(
    #    initial_x=[3,3], N=10, T=500, I=500, max_iters=2000, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=1e-1, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=1e-1, update_clipping=None
    # )
    # problem = CppPaperProblem_ressource_allocation(
    #    initial_x=[0,0, 0], N=10, T=500, I=500, max_iters=2000, lambda_=0.1, GA_steps=300, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-4, update_clipping=5
    # )

    algorithm = Algorithm(problem, verbose=True, empirical_coverage=False, plots=False, zero_order_method=False)

    optimizer = Optimizer(algorithm=algorithm)
    optimizer.run()
    #print(f"{optimizer.result[0]} -> {problem.f_function(optimizer.result[0])} (paper gets 0.1660)")

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
    
