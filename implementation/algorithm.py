import sys
sys.path.append('.')
sys.path.append('..')
import os
import numpy as np
from problem import Problem
from oracle import Oracle
from tqdm import tqdm
from plots import *
from cpp_paper_problem import CppPaperProblem1, CppPaperProblem2

class Algorithm:
    """Algorithm class, containing the main algorithm"""

    def __init__(self, problem: Problem, zero_order_method: bool, empirical_coverage = False, plots: bool = False, verbose: bool = True, abstol = None):#1e-7):
        self.problem = problem
        self.oracle = Oracle(problem)
        self.x = self.problem.initial_x
        self.x_history = np.zeros((self.problem.max_iters, *self.problem.dimension))
        self.objective_function_history = np.zeros(self.problem.max_iters)
        self.chance_constraint_quantile_history = np.zeros(self.problem.max_iters)
        self.empirical_coverage_history = np.zeros(self.problem.max_iters)
        self.feasible_x = []
        self.z_samples = None
        self.zero_order_method = zero_order_method
        self.empirical_coverage = empirical_coverage
        self.plots = plots
        self.verbose = verbose
        self.abstol = abstol
        self.ZO_double_evaluation = False if problem.dimension == (1,) else True 

        if self.problem.unique_chance_constraint_minimizer:
            self.objective_function = lambda x, alpha, S: self.oracle.F_function(x, S)
            # we don't need multiple s samples since there is a unique minimizer
            if self.verbose: tqdm.write(f"We are setting lambda_ to 0 since we do not need to random part of SGLD since there is a unique minimizer, hence we are using GD")
            self.problem.lambda_ = 0
            if self.problem.N != 1:
                if self.verbose: tqdm.write(f"N is not equal to 1, but we are assuming an unique chance constraint minimizer, therefore we set N to 1")
                self.problem.N = 1
            if self.zero_order_method:
                self.x_partial_objective_function = lambda x, alpha, S, z_samples: self.oracle.x_partial_F_function_zero_order(x, alpha, S, z_samples, double_evaluation=self.ZO_double_evaluation)
                if self.verbose: tqdm.write(f"We are using zero order method, therefore there won't be any gradient ascent and the concerned parameters won't be used")
            else:
                self.x_partial_objective_function = lambda x, alpha, S, z_samples: self.oracle.x_partial_F_function(x, S, z_samples)
        else:
            self.objective_function = self.oracle.H_function
            if self.zero_order_method:
                self.x_partial_objective_function = self.oracle.x_partial_H_function_zero_order
            else:
                self.x_partial_objective_function = self.oracle.x_partial_H_function
        if self.zero_order_method:
            if self.ZO_double_evaluation:
                tqdm.write("We will be using double evaluation for our zero order algorithm since the problem is multi-dimentional")
            else:
                tqdm.write("We will not be using double evaluation for our zero order algorithm since the problem is uni-dimentional")
            

    def run(self):

        if self.verbose: tqdm.write(f"The dimension is {self.problem.dimension}")

        for i in tqdm(range(self.problem.max_iters), disable=not self.verbose):

            # checking if the deterministic_constraints function is defined
            if hasattr(self.problem, "deterministic_constraints"):

                # checking if the deterministic_constraints function is satisfied
                if not self.problem.deterministic_constraints(self.x):
                    tqdm.write(f"{self.x} is not feasible since the deterministic constraints are not satisfied")
                    tqdm.write(f"Stopping the algorithm")
                    break


            ## Step 1

            self.z_samples = self.problem.z_samples()

            #if self.verbose: plot_z_samples(self.z_samples)

            if hasattr(self.problem, "compute_states"):
                self.problem.compute_states(self.x, self.z_samples)

            ## Step 2

            initial_s = self.oracle.initialize_s(range=(-100, 100))
            
            final_s, s_history = self.oracle.SGLD(initial_s,
                                       lambda s_t: self.oracle.s_partial_G_function(self.x, s_t, self.z_samples), history=True)
            
            ## Step 3
            max_alpha = None
            # if there is not a unique chance constraint minimizer, we need to find the maximum alpha
            if not self.problem.unique_chance_constraint_minimizer:
                # take the median value of final_s
                #initial_alpha = np.median(final_s)
                initial_alpha = self.problem.initial_alpha

                # gradient ascent
                max_alpha, alpha_history = self.oracle.GA(initial_alpha, 
                                    lambda alpha: self.oracle.alpha_partial_H_function(self.x, alpha, final_s), history=True)
                
            # we verify that the final_s is indeed the superquantile of the chance function
            chance_constraint_quantile, sgld_output_verif, max_alpha_verif = self.oracle.verify_quantiles(self.x, final_s, 1-self.problem.theta_G, self.z_samples, max_alpha)
            #tqdm.write(f"verif: {sgld_output_verif}, {max_alpha_verif}")
            self.chance_constraint_quantile_history[i] = chance_constraint_quantile
            
            if self.plots: 
                s = np.linspace(-5e2, 5e2, 100)
                plot_1D_func(s, func=lambda s: self.oracle.G_function(self.x, s, self.z_samples), title=f"G(x,s;Z) function with fixed x={self.x}", xlabel="s", ylabel="G(x,s;Z)")
                plot_1D_func(s, func=lambda s: self.oracle.s_partial_G_function(self.x, s, self.z_samples), title=f"s_partial_G(x,s;Z) function with fixed x={self.x}", xlabel="s", ylabel="s_partial_G(x,s;Z)")
                # plot H function w.r.t. alpha
                plot_1D_func(s, func=lambda alpha_: self.objective_function(self.x, alpha_, final_s), title=f"H(x,alpha) function with fixed x={self.x}", xlabel="alpha", ylabel="H(x,alpha)")
                # plot partial H function w.r.t. alpha
                plot_s(s_history=s_history)
                if not self.problem.unique_chance_constraint_minimizer:
                    plot_alpha(alpha_history, final_s=final_s)
            
            if chance_constraint_quantile <= 0:
                self.feasible_x.append((self.x, max_alpha, final_s))
                
                if self.verbose: tqdm.write(f"{self.x} -> {self.objective_function(self.x, max_alpha, final_s)} is feasible since the chance constraint quantile is {chance_constraint_quantile}") #self.oracle.H_function(self.x, min_alpha, final_s)
                #self.max_alpha_history.append(True)

            else:
                if self.verbose: tqdm.write(f"{self.x} -> {self.objective_function(self.x, max_alpha, final_s)} is not feasible since the chance constraint quantile is {chance_constraint_quantile}")
                #self.max_alpha_history.append(False)
            
            ## Step 4

            # updating x using the GD algorithm

            self.x_history[i] = self.x
            self.objective_function_history[i] = self.objective_function(self.x, max_alpha, final_s)

            if self.empirical_coverage:
                empirical_coverage = self.oracle.empirical_coverage(self.x, N=500)
                self.empirical_coverage_history[i] = empirical_coverage
                if self.verbose: tqdm.write(f"Empirical coverage: {empirical_coverage}")

            # final_s2 = self.oracle.SGLD(initial_s,
            #                            lambda s_t: self.oracle.s_partial_G_function(self.x, s_t, self.z_samples), history=False)
            
            old_x = self.x

            new_x = self.oracle.GD_one_step(old_x, lambda x: self.x_partial_objective_function(x, alpha=max_alpha, S=final_s, z_samples=self.z_samples)) #0.05

            update = np.linalg.norm(new_x - old_x)

            # check for early convergence
            if self.abstol is not None and update < self.abstol:
                if self.verbose: tqdm.write(f"Early stopping at iteration {i} since the change in x is less than {self.abstol}")
                break

            if self.problem.update_clipping is not None and update > self.problem.update_clipping:
                clipped_update = self.oracle.clip_gradient(update, self.problem.update_clipping)
                if self.verbose: tqdm.write(f"Clipping the update from {update} to {clipped_update}")
                new_x = old_x + clipped_update


            self.x = new_x
                 
            # lambda x: self.oracle.x_partial_H_function4(x, max_alpha, final_s2, self.z_samples, old_x=self.x_history[max(0, i-1)], H_history=self.objective_function_history)

            if self.plots:

                tqdm.write(f"change in x: {self.x - old_x}")

                if self.problem.dimension == 2:

                    func = lambda x_: np.array(self.objective_function(x=x_, alpha=max_alpha, S= final_s))#.mean()
                    x = np.linspace(-5, 5, 20)
                    x, y = np.meshgrid(x, x)
                    X = np.vstack([x.flatten(), y.flatten()]).T
                    plot_2D_func(X, func=func, title=f"objective function function with fixed alpha={max_alpha}")

                    func = lambda x_: np.array(self.problem.partial_f_function(x=x_))
                    plot_2D_func_both(X, func=func, title=f"partial derivate of f(x) w.r.t. x", verbose=False)

                    func = lambda x_: np.array(self.x_partial_objective_function(x=x_, alpha=max_alpha, S= final_s, z_samples=self.z_samples))
                    plot_2D_func_both(X, func=func, title=f"partial derivate of objective function w.r.t. x with fixed alpha={max_alpha}", verbose=True)

                elif self.problem.dimension == 1:
                    
                    x = np.linspace(-5, 5, 100)
                    func = lambda x_: np.array(self.objective_function(x=x_, alpha=max_alpha, S= final_s))
                    plot_1D_func(x, func=func, title=f"objective function function with fixed alpha={max_alpha}")

                    func = lambda x_: np.array(self.problem.partial_f_function(x=x_))
                    plot_1D_func(x, func=func, title=f"partial derivate of f(x) w.r.t. x")

                    func = lambda x_: np.array(self.x_partial_objective_function(x=x_, alpha=max_alpha, S= final_s, z_samples=self.z_samples))
                    plot_1D_func(x, func=func, title=f"partial derivate of the objective function w.r.t. x with fixed alpha={max_alpha}", verbose=True)

            
        # return the feasible x with the minimum f function
        return self.x
    #min(self.feasible_x, key=lambda x: self.problem.f_function(x[0])) if self.feasible_x else None
    

