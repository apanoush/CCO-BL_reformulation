import numpy as np
from scipy.stats import norm

"""
dimentionality of the problem:
- x is a vector of dimension d
- z is a vector of dimension d
- s is a scalar
- alpha is a scalar
"""


class Problem:
    """Problem class, containing all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=1, unique_chance_constraint_minimizer=True, theta_G=0.1, theta_H=1 ,lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping: float=None):
        """
        Args:
            initial_x: initial x value used by the algorithm
            N: sample size of the s ditribution, our auxiliary variable
            T: number of iterations for the SGLD algorithm
            I: number of samples z for the chance constraint
            max_iters: number of iterations for the whole algorithm
            GA_steps: number of steps for the GA algorithm
            unique_chance_constraint_minimizer: if True, we assume that the chance constraint has a unique minimizer if False, we need to find the maximum alpha
            theta_G: 1-theta quantiles; scales down sums in the G function and its partial derivatives
            theta_H: 1-theta quantiles; scales down sums in the H function and its partial derivatives
            lambda_: used in the SGLD algorithm and partial_s function
            learning_rate_SGLD: learning rate for the SGLD algorithm
            learning_rate_GD: learning rate for the GD algorithm
            learning_rate_GA: learning rate for the GA algorithm
            mu: used in the H function
            delta: used in the h function to determine the interval where we approximate the ReLU function
            initial_alpha: initial value for the alpha variable before the GA algorithm
            K: used for the zeroth order H gradient function
            gradient_clipping: if not None, we clip the gradient of GD to this value
        """
        self.initial_x = initial_x
        
        self._compute_dimension()

        self.N = N 
        self.T = T 
        self.I = I
        self.max_iters = max_iters
        self.GA_steps = GA_steps
        self.unique_chance_constraint_minimizer = unique_chance_constraint_minimizer
        self.theta_G = theta_G
        self.theta_H = theta_H
        self.lambda_ = lambda_
        self.learning_rate_SGLD = learning_rate_SGLD
        self.learning_rate_GD = learning_rate_GD
        self.learning_rate_GA = learning_rate_GA
        self.mu = mu
        self.delta = delta
        self.initial_alpha = initial_alpha
        self.K = learning_rate_GD
        self.update_clipping = update_clipping
        self.chance_dimension = ()

        if type(self).__name__ == "Problem":
            self.optimal_solution = 1/(norm.ppf(1-self.theta_G)+1)
            self.optimal_value = self.f_function(self.optimal_solution)

    def _compute_dimension(self):
        if isinstance(self.initial_x, list):
            self.initial_x = np.array(self.initial_x)
            self.dimension = self.initial_x.shape #len(self.initial_x)
        elif isinstance(self.initial_x, np.ndarray):
            self.dimension = self.initial_x.shape
        else:
            self.dimension = (1,)

    def chance_function(self, x, z):
        """chance function, is not deterministic"""
        return x * z -1
    
    def partial_chance_function(self, x, z):
        """differentiation of the chance function with respect to x"""
        return z

    def z_samples(self):
        return np.random.normal(1, 1, self.I)
    
    def h_function(self, x: np.ndarray) -> np.ndarray:
        """relu function"""

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, x, (-1/(16*self.delta**3))*x**4 + (3/(8*self.delta))*x**2 + (1/2)*x + (3/16)*self.delta)
                        )

    def partial_h_function(self, x: np.ndarray) -> np.ndarray:
        """differentiation of the h function"""

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, 1, (-1/(4*self.delta**3))*x**3 + (3/(4*self.delta))*x + (1/2))
            )
    
    def xx_partial_h_function(self, x: np.ndarray) -> np.ndarray:
        """double differentiation of the h function"""
        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, 0, (-3/(4*self.delta**3))*x**2 + (3/(4*self.delta)))
            )

    
    def f_function(self, x):
        return (x-2)**2
    
    def partial_f_function(self, x):
        return 2*x-4
