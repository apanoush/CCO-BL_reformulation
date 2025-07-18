import numpy as np
import sys
sys.path.append('.')
from problem import Problem
from taco_paper_problem import *
from cpp_paper_problem import *
from oracle import Oracle
from algorithm import Algorithm
from plots import *
from cepdd_paper_problem import *

problem_instances = {
    # "Problem": Problem(
    #     initial_x=0.2, N=5, T=300, I=200, max_iters=1000, lambda_=1e-5, GA_steps=1500, theta_G=1-0.95, theta_H=0.1,
    #     learning_rate_SGLD=1e-2, learning_rate_GD=2e-4, learning_rate_GA=2e1, delta=1e-2, mu=1e-2, K=2e-4
    # ),
    "Problem": Problem(
        initial_x=0.1, T=300, N=1, I=500, max_iters=1000, theta_G=1-0.95,
        learning_rate_SGLD=1e-2, learning_rate_GD=2e-4, delta=1e-2, mu=1e-2
    ),
    "TacoPaperProblem2": TacoPaperProblem2(
        initial_x=[0.1]*2, N=10, T=300, I=300, GA_steps=500, max_iters=700, theta_G=1-0.8, learning_rate_SGLD=6e-1, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=2e1, delta= 7e-1
    ),
    "CppPaperProblem2": CppPaperProblem2(
        initial_x=3*np.ones((3, 2)), N=10, T=int(5e4), I=50, max_iters=1000, theta_G=0.05, theta_H=0.2,
        learning_rate_SGLD=7e3, learning_rate_GD=1e-5, delta=2, mu=5, num_robot_steps=2, K=0.5
    ),
    "CppPaperProblem2_2": CppPaperProblem2_2(
        initial_x=3*np.ones((3, 2)),N=1, T=int(5e4), I=1, max_iters=1000, lambda_=0.1, GA_steps=200, theta_G=0.05, theta_H=0.2,
        learning_rate_SGLD=1e-1, learning_rate_GD=1e-9, delta=1e-1, mu=1e2, num_robot_steps=2, K=0.5
    ),
    "CppPaperProblem2_3": CppPaperProblem2_3(
        initial_x=3*np.ones((5, 2)),N=1, T=5, I=1, max_iters=5000, lambda_=0.1, GA_steps=200, theta_G=0.05, theta_H=0.2,
        learning_rate_SGLD=1e-1, learning_rate_GD=1e-3, delta=1e-1, mu=1e2, num_robot_steps=2, K=0.5
    ),
    "TacoPaperProblem1_2": TacoPaperProblem1_2(
       initial_x=[1, -1], N=10, T=2000, I=500, max_iters=200, theta_G=1-0.03368421, learning_rate_SGLD=3e0, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, update_clipping=1/2
    ),
    "TacoPaperProblem1": TacoPaperProblem1(
       initial_x=[0, 0], N=10, T=2000, I=500, max_iters=400, theta_G=1-0.03368421, learning_rate_SGLD=1e0, learning_rate_GD=1e-2, mu=1e-3, delta= 1e-2, update_clipping=3
    ), #I=300
    "TacoPaperProblem1_3": TacoPaperProblem1_3(
       initial_x=[-2, 3], N=10, T=2000, I=500, max_iters=150, theta_G=1-0.03368421, learning_rate_SGLD=3e0, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, update_clipping=1/2
    ),
    "CepddPaperProblem1": CepddPaperProblem1(
       initial_x=np.ones(8)*5, N=10, T=500, I=200, max_iters=150, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
      learning_rate_SGLD=1e-1, learning_rate_GD=1.4e-1, learning_rate_GA=0.1, delta=1e-1, mu=2e0, K=1.4e-2
    ),
    "CepddPaperProblem2": CepddPaperProblem2(
       initial_x=np.ones(9)*5, N=10, T=500, I=200, max_iters=150, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
      learning_rate_SGLD=1e-1, learning_rate_GD=1.4e-1, learning_rate_GA=0.1, delta=1e-1, mu=2e0, K=1.4e-2
    ),
    # "CppPaperProblem1": CppPaperProblem1(
    #     initial_x=-6.35, N=10, T=500, I=500, max_iters=2000, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #     learning_rate_SGLD=1e-1, learning_rate_GD=1e-4, learning_rate_GA=0.1, delta=1e-2, mu=2e0, update_clipping=None
    # ),
    "CppPaperProblem1": CppPaperProblem1(
       initial_x=-5, N=10, T=500, I=500, max_iters=800,
      learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, delta=5e-2, mu=2e0, update_clipping=10
    ),
    # "CppPaperProblem1_0": CppPaperProblem1_0(
    #    initial_x=[3,3], N=10, T=500, I=500, max_iters=2000, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
    #   learning_rate_SGLD=1e-1, learning_rate_GD=1e-1, learning_rate_GA=0.1, delta=1e-2, mu=2e0, update_clipping=None
    # ),
    "CppPaperProblem1_0": CppPaperProblem1_0(
       initial_x=[3.,3.], N=10, T=500, I=500, max_iters=2500,
      learning_rate_SGLD=1e-1, learning_rate_GD=1e-2, delta=1e-2, mu=1, update_clipping=3
    ),
    "CppPaperProblemRAsum": CppPaperProblemRAsum(
       initial_x=[0,0, 0], N=10, T=500, I=500, max_iters=800,
      learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, delta=1e-2, mu=2e0, update_clipping=5
    ),
    "CppPaperProblemRAmax": CppPaperProblemRAmax(
       initial_x=[0,0,0], N=10, T=500, I=500, max_iters=800,
      learning_rate_SGLD=1e-1, learning_rate_GD=3e-4, delta=1e-2, mu=2e0, update_clipping=5
    )
}