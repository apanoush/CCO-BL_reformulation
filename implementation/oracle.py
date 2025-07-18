import numpy as np
import sys
sys.path.append('.')
from problem import Problem
from tqdm import tqdm
import json
import os

class Oracle:
    """Oracle class, containing all the useful functions (e.g. g, h, and the Gibbs measure)"""

    def __init__(self, problem: Problem):
        
        self.problem = problem

    # def objective_function(self, x, alpha, final_s):
    #     """objective function"""

    #     #return self.problem.f_function(x) + self.problem.lambda2 * self.subq_function(alpha, final_s)

    #     return self.problem.f_function(x) + self.problem.lambda2 * self.H_function(alpha, final_s)
    
    # def partial_objective_function(self, x, alpha, S, samples):
    #     """differentiation of the objective function, with respect to x"""
        
    #     #TODO not sure if this is correct (partial_g_function is the differentiation of the g function with respect to s)
    #     return self.problem.partial_f_function(x) + self.problem.lambda2 * self.partial_g_function(x, alpha, samples)

    def G_function(self, x: np.ndarray, s: float, z_samples: np.ndarray) -> float:
        """empirical superquantile convex formulation"""

        sum_ = 0
        for z in z_samples:
            sum_ += self.problem.h_function(self.problem.chance_function(x, z) - s)

        return s + (1/(self.problem.theta_G * self.problem.I)) * sum_
    
    def s_partial_G_function(self, x: np.ndarray, s: float, z_samples) -> float:
        """differentiation of the g function, with respect to s"""

        sum_ = 0

        for z in z_samples:
            sum_ += self.problem.partial_h_function(self.problem.chance_function(x, z) - s)

        return 1 - (1/(self.problem.theta_G * self.problem.I)) * sum_
    
    def x_partial_G_function(self, x: np.ndarray, s: float, z_samples) -> np.ndarray:

        sum_ = 0

        for z in z_samples:
            sum_ += self.problem.partial_h_function(self.problem.chance_function(x, z) - s) * self.problem.partial_chance_function(x, z)

        return -(1/(self.problem.theta_G * self.problem.I)) * sum_
    
    # should be close to 1
    def H_function(self, x, alpha, S) -> np.ndarray:
        """empirical subquantile convex formulation"""
        
        sum_ = 0
        for s in S:
            sum_ += self.problem.h_function(  
                - self.problem.f_function(x) - (s/2) * np.maximum(s/self.problem.mu,0) + alpha 
            )

        return alpha - (1/(self.problem.theta_H * self.problem.N)) * sum_
    
    def alpha_partial_H_function(self, x, alpha, S) -> np.ndarray:
        """differentiation of the subq function, with respect to alpha"""

        sum_ = 0
        for s in S:
            sum_ += self.problem.partial_h_function(  
                - self.problem.f_function(x) - (s/2) * np.maximum(s/self.problem.mu,0) + alpha 
            )

        return 1 - (1/(self.problem.theta_H * self.problem.N)) * sum_
    
    # independent samples from alpha (inner/outer)
    def x_partial_H_function(self, x, alpha, S: np.ndarray, z_samples, split=0.7) -> np.ndarray:
        """differentiation of the subq function, with respect to x"""
        
        sum_ = 0

        # randomly shuffle S to split it into two parts
        S = np.random.permutation(S)

        S1 = S[:int(split*S.shape[0])]
        S2 = S[int(split*S.shape[0]):]

        for s1 in S1:
            sum_ += self.problem.partial_h_function(
                -self.problem.f_function(x) - (s1/2) * np.maximum(s1/self.problem.mu,0) + alpha 
            ) * self.problem.partial_f_function(x) - self.problem.h_function(
                -self.problem.f_function(x) - (s1/2) * np.maximum(s1/self.problem.mu,0) + alpha 
            )  * self.j_part(x, S2, s1, z_samples)

        return (1/(self.problem.theta_H * len(S1))) * sum_
    
    def j_part(self, x, S2, s1, z_samples):

        sum_ = 0
        for s2 in S2:
            sum_ += self.x_partial_G_function(x, s2, z_samples)

        return (1/self.problem.lambda_) * ( (1/len(S2)) * sum_ - self.x_partial_G_function(x, s1, z_samples) )

    # def gibbs_measure(self, x, s, samples):
    #     """Gibbs measure"""

    #     pass

    #     return np.exp((-1/self.problem.lambda_) * self.g_function(x, s, samples))

    #     param = (-1/self.problem.lambda_) * self.g_function(x, s, samples)

    #     return  param * np.exp (-param)
    
    def initialize_s(self, range: tuple[int, int]=(-1, 1)) -> np.ndarray:
        """initialize s, the starting point of the SGLD algorithm"""
        
        return np.linspace(*range, self.problem.N) if self.problem.N > 1 else sum(range)//2
    
        #return np.random.uniform(*range, (self.problem.N, self.problem.dimension))
    
    def SGLD(self, s_0: np.ndarray, func, seed=None, history = False, abstol=1e-3) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Stochastic Gradient Langevin Descent algorithm"""

        if seed is not None:
            np.random.seed(seed)

        s_t = s_0

        if history: 
            s_history = np.zeros((self.problem.N, self.problem.T+1, *self.problem.chance_dimension))
            s_history[:, 0] = s_t

        for i in range(1, self.problem.T + 1):
            prev_s = s_t
            s_t = s_t - self.problem.learning_rate_SGLD * func(s_t) + np.sqrt(self.problem.learning_rate_SGLD * self.problem.lambda_) * np.random.randn()

            if history: s_history[:, i] = s_t

            # check for early convergence
            if abstol is not None and np.linalg.norm(s_t - prev_s) < abstol:
                break
        
        if history: return s_t, s_history
        else: return s_t, None
    
    def GD_one_step(self, x, func):
        """one step Gradient Descent algorithm"""
        gradient = func(x)

        if self.problem.update_clipping:
            gradient = self.clip_gradient(gradient, self.problem.update_clipping)

        return x - self.problem.learning_rate_GD * gradient
    
    def GA(self, initial_x, func, history=False):
        """Gradient Ascent algorithm"""

        alpha_history = np.zeros(self.problem.GA_steps+1)

        x = initial_x
        alpha_history[0] = x


        for i in range(1,self.problem.GA_steps+1):
            x = x + self.problem.learning_rate_GA * func(x)
            #print("ooook", x, type(x))

            alpha_history[i] = x

        if history:
            return x, alpha_history
        else:
            return x
        

    def Adam(self, initial_x: np.ndarray, func, history: bool = False, 
             beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Adam optimization algorithm.

        Parameters:
            initial_x: starting parameter vector (or array).
            func: function returning the gradient at x.
            N_steps: number of optimization steps.
            history: if True, keep track of x updates.
            beta1: exponential decay rate for the first moment estimate.
            beta2: exponential decay rate for the second moment estimate.
            epsilon: small constant for numerical stability.

        Returns:
            The optimized x and optionally the history array.
        """
        x = initial_x.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        if history:
            x_history = np.zeros((self.problem.N, self.problem.T+1))
            x_history[:,0] = x

        # Using a learning rate defined in problem if available, else default to 0.001
        for t in range(1, self.problem.T + 1):
            grad = func(x)

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            x = x - self.problem.learning_rate_SGLD * m_hat / (np.sqrt(v_hat) + epsilon) + np.sqrt(self.problem.learning_rate_SGLD * self.problem.lambda_) * np.random.randn()

            if history:
                x_history[:,t] = x

        return (x, x_history) if history else x
        

    def x_partial_H_function3(self, x, alpha, S: np.ndarray, z_samples, split=0.5) -> np.ndarray:
        """Differentiation of H_function with respect to x,
        including the effect of the s-distribution variation via partial_s.
        
        Instead of duplicating h'(.), we combine the derivative contributions
        from f(x) and s.
        """
        sum_ = 0

        S1 = S[:int(split * S.shape[0])]
        S2 = S[int(split * S.shape[0]):]

        for s1 in S1:
            # Compute the inner argument used in h
            inner = -self.problem.f_function(x) - (s1/2) * np.maximum(s1/self.problem.mu, 0) - alpha
            # The derivative picked via the chain rule has two contributions:
            #   (i) from f:   -f'(x)  => contributes +f'(x) after returning to H
            #   (ii) from s:   derivative of s-part wrt x, computed via partial_s.
            deriv_inner = self.problem.partial_f_function(x) + self.j_part(x, S2, s1, z_samples)
            sum_ += self.problem.partial_h_function(inner) * deriv_inner

        return (1 / (self.problem.theta * len(S1))) * sum_
    
    def x_partial_H_function2(self, x, alpha, S) -> np.ndarray:
        """Partial derivative of H_function with respect to x."""
        sum_deriv = 0
        for s in S:
            # Compute the inner argument for h
            inner = - self.problem.f_function(x) - (s/2) * np.maximum(s/self.problem.mu, 0) - alpha
            # Chain rule: derivative is h'(inner)*(- f'(x)), then the pre-factor cancels the minus sign
            sum_deriv += self.problem.partial_h_function(inner) * self.problem.partial_f_function(x)
        return (1 / (self.problem.theta * self.problem.N)) * sum_deriv
    
    def H_function4(self, x, alpha, S) -> np.ndarray:
        """empirical subquantile convex formulation"""
        
        sum_ = 0
        for s in S:
            sum_ += self.problem.h_function(  
                - self.problem.f_function(x) - self.min_s(x) + alpha 
            )

        return alpha - (1/(self.problem.theta_H * self.problem.N)) * sum_
    
    def min_s(self, x):
        """minimum of the G(x,s,Z) function w.r.t s"""
        s_0 = 0
        return self.SGLD(s_0, lambda s: self.s_partial_G_function(x, s, self.problem.z_samples()))
    
    def x_partial_H_function_zero_order(self, x , alpha, S, z_samples,old_x, H_history) -> np.ndarray:
        """differentiation of the subq function, with respect to x"""

        # randomly select between points from the self.dimension-unit sphere
        if self.problem.dimension == 1:
            z_from_unit_circle = np.random.choice([-1, 1])
        else:
            z_from_unit_circle = np.random.randn(self.problem.dimension)
            z_from_unit_circle /= np.linalg.norm(z_from_unit_circle)

        #tqdm.write(f"x: {x}, old_x: {old_x}")
        previous_step_difference = np.abs(np.array(x)-np.array(old_x))
        momentum = previous_step_difference/(previous_step_difference + 1e-1) + 1e-1 #1e-2
        tqdm.write(f"momentum: {momentum}")

        self.problem.K = z_from_unit_circle * self.problem.K * momentum

        # we recompute SGLD for new x+k
        initial_s = self.initialize_s(range=(-100, 100))
        alpha_0 = self.problem.initial_alpha
        x_2 = x + self.problem.K
        S_2, _ = self.SGLD(initial_s, lambda s_t: self.s_partial_G_function(x_2, s_t, z_samples), history=False)
        max_alpha_2, _ = self.GA(alpha_0, lambda alpha_: self.alpha_partial_H_function(x_2, alpha_, S_2), history=True)

        zero_order_estimation = self.H_function(x_2, max_alpha_2, S_2) - self.H_function(x, alpha, S)
        gradient_estimation = (z_from_unit_circle * zero_order_estimation) / (self.problem.K + 1e-2) 

        if self.H_function(x_2, max_alpha_2, S_2) > 2 * np.array(H_history).mean():
            return np.zeros(self.problem.dimension) if self.problem.dimension > 1 else 0

        return gradient_estimation
        #return np.clip(gradient, -1000, 1000)

    def F_function(self, x: np.ndarray, min_s: float) -> float:
        """F function"""
        return self.problem.f_function(x) + min_s * np.maximum(min_s/self.problem.mu, 0)
    
    def x_partial_F_function(self, x: np.ndarray, min_s: float, z_samples) -> np.ndarray:
        """Differentiation of F function with respect to x."""
        return self.problem.partial_f_function(x) + 2 * self.x_partial_s(x, min_s, z_samples) * np.maximum(min_s/self.problem.mu, 0) 
    
    def x_partial_s(self, x, s, z_samples) -> np.ndarray:
        """Partial derivative of s."""
        return -self.sx_partial_G_function(x, s, z_samples) / (self.ss_partial_G_function(x, s, z_samples) + 1e-10)
    
    def sx_partial_G_function(self, x, s, z_samples) -> np.ndarray:
        """Partial derivative of G function with respect to s and x"""

        sum_ = 0
        for z in z_samples:
            sum_ +=  self.problem.xx_partial_h_function(self.problem.chance_function(x, z) - s) * self.problem.partial_chance_function(x, z) # np.dot ?

        return - 1 / (self.problem.theta_G * self.problem.I) * sum_
    
    def ss_partial_G_function(self, x, s, z_samples) -> np.ndarray:
        """Partial derivative of G function with respect to s and s"""

        sum_ = 0
        for z in z_samples:
            sum_ += self.problem.xx_partial_h_function(self.problem.chance_function(x, z) - s)

        return 1 / (self.problem.theta_G * self.problem.I) * sum_
    
    def x_partial_F_function_zero_order(self, x , alpha, S, z_samples, double_evaluation= True, random_K_amplificator=True) -> np.ndarray:
        """differentiation of the F function, with respect to x"""
        M = 3/2#5
        random_K_amplificator = np.random.uniform(1/M, M) if random_K_amplificator else 1

        # randomly select between points from the self.dimension-unit sphere
        if self.problem.dimension == 1:
            z = [np.random.choice([-1, 1])]
        else:
            z_from_unit_circle = np.random.randn(*self.problem.dimension)
            z_from_unit_circle /= np.linalg.norm(z_from_unit_circle)
            z = [z_from_unit_circle]
            if double_evaluation:
                # generate a random vector and project out its component along z_from_unit_circle
                rand_vec = np.random.randn(z_from_unit_circle.shape[0])
                proj = np.dot(rand_vec, z_from_unit_circle)
                z_perp = rand_vec - proj * z_from_unit_circle
                # normalize to unit length
                z_perp /= (np.linalg.norm(z_perp) + 1e-5)
                z.append(z_perp)

            zero_order_estimations = []

            for z_ in z:

                perturbation = z_ * self.problem.K * random_K_amplificator
                # we recompute GD for new x+perturbation
                initial_s = 0
                x_plus = x + perturbation
                S_plus, _ = self.SGLD(initial_s, lambda s_t: self.s_partial_G_function(x_plus, s_t, z_samples), history=False)
                x_minus = x - perturbation
                S_minus, _ = self.SGLD(initial_s, lambda s_t: self.s_partial_G_function(x_minus, s_t, z_samples), history=False)

                zero_order_estimation = z_ * (self.F_function(x_plus, S_plus) - self.F_function(x_minus, S_minus)) / (2 * self.problem.K + 1e-5)
                zero_order_estimations.append(zero_order_estimation)

        return np.array(zero_order_estimations).mean(axis=0)

    def verify_quantiles(self, x, sgld_output: np.ndarray, p, z_samples, max_alpha=None ) -> tuple[float, float, float]:
        """verify that the output of the SGLD algorithm, and max_alpha is empirically correct
        Parameters:
        - x: current value of x
        - sgld_output: output of the SGLD algorithm
        - max_alpha: output of the GA algorithm
        - p: quantile level
        - z_samples: samples of z
        """

        # we sample the chance_constraint with our z_samples
        samples = np.zeros((len(z_samples), *self.problem.chance_dimension))
        for i, z in enumerate(z_samples):
            samples[i] = self.problem.chance_function(x, z)

        # we compute the empirical p-quantile of this chance_constraint function
        empirical_p_quantile = np.percentile(samples, 100*p)

        # we take the 1-p quantile of our SGLD output 
        sgld_output_1p_quantile = np.percentile(sgld_output, 100*(1-p))

        # we check if the superquantile is correct
        
        max_alpha_check = max_alpha - empirical_p_quantile if max_alpha is not None else None

        return empirical_p_quantile, sgld_output_1p_quantile - empirical_p_quantile, max_alpha_check

    def empirical_coverage(self, x, N=5000):
        """computes the empirical coverage of the chance constraint, i.e. frequency for with it is satisfied for a fixed x

        Args:
            x (np.ndarray): x value
            N (int, optional): number of samples. Defaults to 5000.
        """
        samples = np.zeros((N, self.problem.I))
        for i in range(N):
            z_samples = self.problem.z_samples()
            for j, z in enumerate(z_samples):
                samples[i,j] = True if self.problem.chance_function(x, z) <= 0 else False
        # return percentage of samples for which the chance constraint is satisfied
        return np.mean(samples.flatten())
    
    def save_results(self, x, f_value, empirical_coverage, problem_name):
        json_path = os.path.join(os.path.dirname(__file__), "results.json")
        results = json.load(open(json_path, "r"))
        results = {} if results is None else results
        results[problem_name] = {
            "x": x.tolist() if isinstance(x, np.ndarray) else x,
            "f_value": f_value,
            "empirical_coverage": empirical_coverage
        }
        json.dump(results, open(json_path, "w"), indent=4)

    @staticmethod
    def suboptimality_f(problem: Problem, x_history) -> np.ndarray:
        return np.array([abs(problem.f_function(x) - problem.optimal_value) / abs(problem.optimal_value) for x in x_history])
    
    @staticmethod
    def supoptimality_ec(problem: Problem, empirical_coverage_history) -> np.ndarray:
        P = 1-problem.theta_G
        return np.array([(ec - P) / abs(P) for ec in empirical_coverage_history])
    
    @staticmethod
    def clip_gradient(gradient, max_norm):
        """
        Clips a single gradient tensor to a maximum norm.
        
        Args:
            gradient (np.ndarray): Gradient tensor to clip.
            max_norm (float): Maximum allowed L2 norm for this gradient.
        
        Returns:
            np.ndarray: Clipped gradient.
        """
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > max_norm:
            gradient = gradient * (max_norm / grad_norm)
        return gradient


# def x_partial_H_function_zero_order(self, x , alpha, S, z_samples,old_x, H_history) -> np.ndarray:
#         """differentiation of the subq function, with respect to x"""
        
#         # res = np.zeros(K)
#         # for k in range(K):
#         #     res[k] = self.H_function4(x, alpha, S) - self.H_function4(x+1, alpha, S)
#         # return res.mean()

#         # randomly select between points from the self.dimension-unit sphere
#         if self.problem.dimension == 1:
#             z_from_unit_circle = np.random.choice([-1, 1])
#         else:
#             z_from_unit_circle = np.random.randn(self.problem.dimension)
#             z_from_unit_circle /= np.linalg.norm(z_from_unit_circle)

#         #tqdm.write(f"x: {x}, old_x: {old_x}")
#         previous_step_difference = np.abs(np.array(x)-np.array(old_x))
#         momentum = previous_step_difference/(previous_step_difference + 1e-1) + 1e-1 #1e-2
#         tqdm.write(f"momentum: {momentum}")

#         self.problem.K = z_from_unit_circle * self.problem.K * momentum

#         # we recompute SGLD for new x+k
#         initial_s = self.initialize_s(range=(-100, 100))
#         alpha_0 = self.problem.initial_alpha
#         x_2 = x + self.problem.K
#         S_2 = self.SGLD(initial_s, lambda s_t: self.s_partial_G_function(x_2, s_t, z_samples), history=False)
#         max_alpha_2, _ = self.GA(alpha_0, lambda alpha_: self.alpha_partial_H_function(x_2, alpha_, S_2), history=True)

#         zero_order_estimation = self.H_function(x_2, max_alpha_2, S_2) - self.H_function(x, alpha, S)
#         gradient_estimation = (z_from_unit_circle * zero_order_estimation) / (self.problem.K + 1e-2) 

#         if self.H_function(x_2, max_alpha_2, S_2) > 2 * np.array(H_history).mean():
#             return np.zeros(self.problem.dimension) if self.problem.dimension > 1 else 0

#         return gradient_estimation