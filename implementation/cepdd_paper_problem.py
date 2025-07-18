import sys
sys.path.append('.')
from problem import Problem
import numpy as np
from tqdm import tqdm
from scipy.stats import expon

"""
in the paper the number of z samples I is called K (L for their "validation" set)
V is the size for "additional test dataset" (page 10)

our algorithm converges to x=-3
"""

class CepddPaperProblem1(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=1-0.9, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K)
        self.optimal_solution = -np.log(10) -np.log(expon.ppf(1-self.theta_G, scale=3))
        self.optimal_value = self.f_function(self.optimal_solution)
        self.dimension = 8
        
    
        m = 4
        n = 2*m

        self.chance_dimension = (m,m)

        # after m = 4; n = 2*m

        # build a list of n permutation‐matrices:
        #   the first is I_n,
        #   the second is I_n with its first row moved to the bottom,
        #   the third with the second row moved to the bottom, etc.

        self.permutation_matrices = np.stack([
            np.roll(np.eye(m), -i, axis=0)
            for i in range(n)
        ], axis=0)

        self.expected_demand = np.array([
            [2, 1, 3, 4],
            [2, 3, 2, 1],
            [1, 2, 4, 2],
            [3, 2, 1, 3]
        ])

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        # use the expected_demand array as the λ‐parameter for a Poisson draw
        shape = self.expected_demand.shape
        return np.random.poisson(lam=self.expected_demand, size=shape)
    
    def deterministic_constraints(self, x) -> bool:
        return all(x >= 0)

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return - (self.permutation_matrices.T @ x - z)#.flatten().sum()

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
        return - self.permutation_matrices#.flatten().sum()

    def h_function(self, x: np.ndarray) -> np.ndarray:
        """relu function"""

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, x, (-1/(16*self.delta**3))*x**4 + (3/(8*self.delta))*x**2 + (1/2)*x + (3/16)*self.delta)
                        )

    def partial_h_function(self, x: np.ndarray) -> np.ndarray:
        """differentiation of the h function"""
        # differentiation of the ReLU function

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, 1, (-1/(4*self.delta**3))*x**3 + (3/(4*self.delta))*x + (1/2))
            )

    def f_function(self, x):
        return np.sum(x)

    def partial_f_function(self, x) -> np.ndarray:
        return np.ones(self.dimension)


class CepddPaperProblem2(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)
    num_robot_steps is T in paper, number of steps for the robot
    I is always 4
    self.x is here the self.u_t_history ndarray of size (K, 2)
    """

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=1-0.9, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, num_robot_steps=10, zeta=1, K=1):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K=1)

        self.y_0 = np.array([0, 0, 0, 0], dtype=np.float64)
        self.y_t = self.y_0
        #assert self.initial_x.shape == (num_robot_steps, 2), "initial_x should be of shape (num_robot_steps, 2)"
        if self.initial_x.shape != (num_robot_steps, 2):
            tqdm.write(f"initial_x shape: {self.initial_x.shape} has not a correct shape, it should have the same shape as num_robot_steps: {num_robot_steps}")
            tqdm.write(f"setting initial_x to ones of shape (num_robot_steps, 2)")
            self.initial_x = np.ones((num_robot_steps, 2), dtype=np.float64)
            self._compute_dimension()
        self.target_location = np.array([5, 5])
        self.num_robot_steps = num_robot_steps
        self.y_K = None
        self.zeta = zeta
        self.K = K

        self.expected_demand = np.array([2, 3, 2, 1, 4, 2, 3, 2, 4, 3, 2, 1, 3, 2])
        self.c = np.array([
            10, 15, 18, 15,    # Routes 0-3
            32, 32, 57, 57,    # Routes 4-7
            60, 60, 63, 63,    # Routes 8-11
            61, 61, 75, 75,    # Routes 12-15
            62, 62, 44         # Routes 16-18
        ])
        self.arcs = np.array([
            [1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0],
            [0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,1,0],
            [0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0],
            [0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0],
            [0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0],
            [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0],
            [0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,1,0],
            [0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,1,1,0],
            [0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1],
            [0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1]
        ])
        self.chance_dimension = [14]
        self.initial_x = np.ones(19)*5
        print(self.initial_x)

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        shape = self.expected_demand.shape
        return np.random.poisson(lam=self.expected_demand, size=shape)

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return -(np.dot(self.arcs, x) - z)

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
        return -self.arcs

    def h_function(self, x: np.ndarray) -> np.ndarray:
        """relu function"""

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, x, (-1/(16*self.delta**3))*x**4 + (3/(8*self.delta))*x**2 + (1/2)*x + (3/16)*self.delta)
                        )

    def partial_h_function(self, x: np.ndarray) -> np.ndarray:
        """differentiation of the h function"""
        # differentiation of the ReLU function

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, 1, (-1/(4*self.delta**3))*x**3 + (3/(4*self.delta))*x + (1/2))
            )

    def f_function(self, x: np.ndarray) -> float:
        return np.dot(self.c, x)

    def partial_f_function(self, x: np.ndarray) -> np.ndarray:
        return self.c
   
