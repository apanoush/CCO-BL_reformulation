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

class CppPaperProblem1_0(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=1-0.9, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping=None):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K, update_clipping=update_clipping)
        self.optimal_solution = None
        self.optimal_value = None
        self.theta_G = 0.1
        self.c = np.array([-1,-2])

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        return np.random.uniform(15,16, self.I)

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return (x[0]-3)**2 + (x[1]-5)**2 - z

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
        return np.array([2*(x[0]-3), 2*(x[1]-5)])

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
        return np.dot([-1,-2], x)

    def partial_f_function(self, x) -> np.ndarray:
        return np.array([-1,-2])

class CppPaperProblem1(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.1, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping=None):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K, update_clipping=update_clipping)
        self.optimal_solution = -np.log(10) -np.log(expon.ppf(1-self.theta_G, scale=20))
        self.optimal_value = self.f_function(self.optimal_solution)
        self.theta_G = 0.1

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        return np.random.exponential(20, self.I) #3
    
    def deterministic_constraints(self, x) -> bool:
        return x**3 + 20 <= 0

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return 50 * z * np.exp(x) - 5

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
        return 50 * z * np.exp(x)

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
        return x**3 * np.exp(x)

    def partial_f_function(self, x) -> np.ndarray:
        return 3 * x**2 * np.exp(x) + x**3 * np.exp(x)


class CppPaperProblem2(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)
    num_robot_steps is T in paper, number of steps for the robot
    I is always 4
    self.x is here the self.u_t_history ndarray of size (K, 2)
    """

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.05, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, num_robot_steps=10, zeta=1, K=1):
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

        self.A = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]])
        self.B = np.array([
            [0.5, 0],
            [1, 0],
            [0, 0.5],
            [0, 1]])
        
        self.C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])


class CppPaperProblem2_2(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)
    num_robot_steps is T in paper, number of steps for the robot
    I is always 4
    self.x is here the self.u_t_history ndarray of size (K, 2)
    """

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.05, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, num_robot_steps=10, zeta=1, K=1):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K=1)

        self.y_0 = np.array([0, 0, 0, 0], dtype=np.float64)
        self.y_t = self.y_0
        #assert self.initial_x.shape == (num_robot_steps, 2), "initial_x should be of shape (num_robot_steps, 2)"
        if self.initial_x.shape != (num_robot_steps, 2):
            tqdm.write(f"initial_x shape: {self.initial_x.shape} has not a correct shape, it should have the same shape as num_robot_steps: {num_robot_steps}")
            tqdm.write(f"setting initial_x to ones of shape (num_robot_steps, 2)")
            self.initial_x = np.ones((num_robot_steps, 2), dtype=np.float64)
            self._compute_dimension()
        self.target_location = np.array([3.5, 3.5])
        self.num_robot_steps = num_robot_steps
        self.y_K = None
        self.zeta = zeta
        self.K = K
        self.B = np.array([
            [0.5, 0],
            [1, 0],
            [0, 0.5],
            [0, 1]])
        
        self.C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])



    def compute_states(self, x: np.ndarray, z_samples):
        """compute the states of the robot"""

        assert np.array(x).shape == (self.num_robot_steps, 2), "initial_x should be of shape (num_robot_steps, 2)"
  
        for k in range(self.num_robot_steps):
            self.y_t = self.y_t + self.B @ x[k] + z_samples[k]

        self.y_K = self.y_t

        return None

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        return np.ones((self.num_robot_steps, 4))
        return np.random.laplace(0, 0.02, (self.num_robot_steps, 4)) / 4

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        assert self.y_K is not None, "compute_states should be called before chance_function"
        return (self.y_K[0] - self.target_location[0])**2 + (self.y_K[2] - self.target_location[1])**2 - self.zeta
        #return ((self.C @ self.y_K - self.target_location)**2).sum() - self.zeta

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
  
        #grad_u = np.ones_like(x) * (x.flatten().sum()/2 + z[0] + z[2]  - self.target_location.sum() )
        grad_u = np.ones_like(x) * (self.y_K[0] + self.y_K[2]  - self.target_location.sum() )

        return grad_u



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

    def f_function(self, u_t_history: np.ndarray) -> float:
        #return np.sum(u_t_history**2, axis=1).sum()
        return np.sum(np.array(u_t_history).flatten()**2)

    def partial_f_function(self, u_t_history: np.ndarray) -> np.ndarray:
        return 2 * u_t_history

class CppPaperProblem2_3(Problem):
    
    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.05, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, num_robot_steps=10, zeta=1, K=1, delta_cc=0.1):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K=1)
        self.y_0 = np.array([0, 0, 0, 0], dtype=np.float64)  # [x1, v1, x2, v2]
        self.y_t = self.y_0.copy()
        self.target_location = np.array([5.0, 5.0])
        self.T = T  # Time horizon
        self.zeta = zeta
        self.delta_cc = delta_cc  # Chance constraint delta
        
        # Dynamics matrices (A is identity as simplified)
        self.A = np.eye(4)
        self.B = np.array([
            [0.5, 0],
            [1,   0],
            [0, 0.5],
            [0,   1]
        ])
        
        # Position extraction matrix
        self.C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        
        # Initialize control sequence
        if initial_x.shape != (T, 2):
            print(f"Reshaping control sequence to ({T}, 2)")
            self.u = np.zeros((T, 2), dtype=np.float64)
        else:
            self.u = initial_x.copy()
        
        # Store state history
        self.state_history = np.zeros((T+1, 4))
        self.state_history[0] = self.y_0

    def reset_state(self):
        self.y_t = self.y_0.copy()
        self.state_history[0] = self.y_0

    def compute_states(self, u_sequence, noise_sequence):
        """Compute state evolution with control and noise"""
        self.reset_state()
        y = self.y_0.copy()
        
        for t in range(self.T):
            # y_{t+1} = Ay_t + Bu_t + w_t
            y = self.A @ y + self.B @ u_sequence[t] + noise_sequence#[t]
            self.state_history[t+1] = y
        
        self.y_t = y  # Final state
        return y

    def generate_noise(self):
        """Generate Laplace noise for T steps"""
        # Scale parameter adjusted for 4D noise
        return np.random.laplace(scale=0.02, size=(self.T, 4)) / 4

    def chance_function(self, x, noise_sequence=None):
        """Calculate constraint: P(|position - target|² ≤ ζ) ≥ 1-δ"""
        if noise_sequence is None:
            noise_sequence = self.generate_noise()
            
        final_state = self.compute_states(self.u, noise_sequence)
        final_pos = self.C @ final_state
        distance_sq = np.sum((final_pos - self.target_location)**2)
        return distance_sq - self.zeta

    def partial_chance_function(self, x, noise_sequence):
        """Gradient of chance constraint w.r.t control inputs"""
        self.compute_states(self.u, noise_sequence)
        final_pos = self.C @ self.y_t
        pos_error = final_pos - self.target_location
        
        # Gradient calculation through time
        grad = np.zeros((self.T, 2))
        for t in range(self.T):
            # Gradient propagation: ∂y_T/∂u_t = Bᵀ * (Aᵀ)^{T-t-1} * Cᵀ
            # Since A is identity, this simplifies to BᵀCᵀ
            grad_component = self.B.T @ self.C.T @ (2 * pos_error)
            grad[t] = grad_component
            
        return grad

    def f_function(self, u):
        """Sum of squared controls: Σ||u_t||²"""
        return np.sum(u**2)

    def partial_f_function(self, u):
        """Gradient of objective: 2*u_t"""
        return 2 * u

    def solve(self, learning_rate=0.01, iterations=1000):
        """Basic gradient-based solver"""
        for i in range(iterations):
            # Generate new noise sample each iteration
            noise = self.generate_noise()
            
            # Compute gradients
            grad_obj = self.gradient_objective()
            grad_constraint = self.gradient_chance_constraint(noise)
            
            # Update control sequence
            self.u -= learning_rate * (grad_obj + grad_constraint)
            
            # Projection to feasible set (if needed)
            # self.u = np.clip(self.u, -1, 1)  # Example constraint
            
            # Monitor progress
            if i % 100 == 0:
                obj_val = self.objective_function()
                constr_val = self.chance_constraint()
                print(f"Iter {i}: Obj={obj_val:.4f}, Constr={constr_val:.4f}")
        
        return self.u

class CppPaperProblem2_4(Problem):
    
    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, 
                 theta_G=0.05, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, 
                 learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, num_robot_steps=10, zeta=1, K=1):
        
        # Initialize parent class parameters (assuming they exist)
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
        self.K = K
        self.chance_dimension = (1,)

        # Robot-specific parameters
        self.y_0 = np.array([0, 0, 0, 0], dtype=np.float64)
        self.num_robot_steps = num_robot_steps
        self.target_location = np.array([5, 5])  # From paper: target at (5,5)
        self.zeta = zeta
        
        # Validate and set initial control sequence
        if initial_x.shape != (num_robot_steps, 2):
            tqdm.write(f"initial_x shape: {initial_x.shape} incorrect, should be ({num_robot_steps}, 2)")
            tqdm.write(f"Setting initial_x to zeros of shape ({num_robot_steps}, 2)")
            self.initial_x = np.zeros((num_robot_steps, 2), dtype=np.float64)
        else:
            self.initial_x = initial_x
            
        self._compute_dimension()
        
        # System matrices from the paper
        self.A = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.B = np.array([
            [0.5, 0],
            [1, 0],
            [0, 0.5],
            [0, 1]
        ], dtype=np.float64)
        
        # Output matrix to extract positions
        self.C = np.array([
            [1, 0, 0, 0],  # x^(1)
            [0, 0, 1, 0]   # x^(2)
        ], dtype=np.float64)
        
        # State trajectory storage
        self.y_trajectory = None
        self.y_final = None

    def compute_states(self, u: np.ndarray, w_samples: np.ndarray):
        """
        Compute the state trajectory using discrete-time dynamics:
        y_{t+1} = A*y_t + B*u_t + w_t
        """
        assert u.shape == (self.num_robot_steps, 2), f"u should be shape ({self.num_robot_steps}, 2)"
        assert w_samples.shape == (self.num_robot_steps, 4), f"w_samples should be shape ({self.num_robot_steps}, 4)"
        
        # Initialize trajectory storage
        self.y_trajectory = np.zeros((self.num_robot_steps + 1, 4))
        self.y_trajectory[0] = self.y_0.copy()
        
        # Simulate the system dynamics
        for t in range(self.num_robot_steps):
            self.y_trajectory[t + 1] = (self.A @ self.y_trajectory[t] + 
                                      self.B @ u[t] + 
                                      w_samples[t])
        
        # Store final state
        self.y_final = self.y_trajectory[-1].copy()
        
        return self.y_trajectory

    def z_samples(self):
        """Generate noise samples w_t for the system"""
        # Return noise samples from predefined distribution
        # Using small Laplace noise as mentioned in your original code
        return np.random.laplace(0, 0.02, (self.num_robot_steps, 4)) / 4

    def chance_function(self, u: np.ndarray, w: np.ndarray) -> float:
        """
        Chance constraint function: (x_T^(1) - 5)^2 + (x_T^(2) - 5)^2 - ζ
        Returns the constraint violation (should be ≤ 0 with high probability)
        """
        # Compute states with given control and noise
        self.compute_states(u, w)
        
        # Extract final position using output matrix C
        final_position = self.C @ self.y_final
        
        # Compute distance squared to target minus threshold
        distance_sq = np.sum((final_position - self.target_location)**2)
        
        return distance_sq - self.zeta

    def partial_chance_function(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Gradient of chance function with respect to control u
        Using chain rule: ∂g/∂u = ∂g/∂y_T * ∂y_T/∂u
        """
        # Ensure states are computed
        self.compute_states(u, w)
        
        # Gradient of constraint w.r.t. final state
        final_position = self.C @ self.y_final
        grad_g_y = 2 * self.C.T @ (final_position - self.target_location)  # Shape: (4,)
        
        # Compute ∂y_T/∂u using backward propagation
        grad_u = np.zeros_like(u)  # Shape: (num_robot_steps, 2)
        
        # Backward pass through the dynamics
        grad_y = grad_g_y.copy()  # Gradient w.r.t. y_T
        
        for t in reversed(range(self.num_robot_steps)):
            # Gradient w.r.t. u_t
            grad_u[t] = self.B.T @ grad_y
            
            # Propagate gradient backward through A
            grad_y = self.A.T @ grad_y
        
        return grad_u

    def h_function(self, x: np.ndarray) -> np.ndarray:
        """Smooth approximation of ReLU function"""
        return np.where(x <= -self.delta, 0, 
                       np.where(x >= self.delta, x, 
                               (-1/(16*self.delta**3))*x**4 + 
                               (3/(8*self.delta))*x**2 + 
                               (1/2)*x + (3/16)*self.delta))

    def partial_h_function(self, x: np.ndarray) -> np.ndarray:
        """Derivative of smooth ReLU approximation"""
        return np.where(x <= -self.delta, 0, 
                       np.where(x >= self.delta, 1, 
                               (-1/(4*self.delta**3))*x**3 + 
                               (3/(4*self.delta))*x + (1/2)))

    def f_function(self, u: np.ndarray) -> float:
        """
        Objective function: sum of squared control inputs
        ∑_{t=0}^{T-1} ||u_t||_2^2
        """
        return np.sum(u**2)

    def partial_f_function(self, u: np.ndarray) -> np.ndarray:
        """Gradient of objective function"""
        return 2 * u

class CppPaperProblem3(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.05, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K)

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        return np.random.exponential(1/3, self.I)
    
    def deterministic_constraints(self, x) -> bool:
        return x**3 + 20 <= 0

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return 50 * z * np.exp(x) - 5

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
        return 50 * z * np.exp(x)

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
        return x**3 * np.exp(x)

    def partial_f_function(self, x) -> np.ndarray:
        return 3 * x**2 * np.exp(x) + x**3 * np.exp(x)

class CppPaperProblemRAsum(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.1/(3**2), theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping=None):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K, update_clipping=update_clipping)
        self.optimal_solution = None
        self.optimal_value = None
        self.objective_vector = np.array([1,1,1])
        self.constraint_matrix = np.array([[3,12,2],[10,3,5], [5,3,15]])
        self.theta_G = 0.1#/(3)
        tqdm.write(f"theta_G: {self.theta_G}")

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        return np.random.lognormal(0,0.5**2, (self.I, *self.dimension))

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return (z-np.dot(self.constraint_matrix, x)).max()#.sum()

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
        return -self.constraint_matrix.sum(axis=0)

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
        return np.dot(self.objective_vector, x)

    def partial_f_function(self, x) -> np.ndarray:
        return self.objective_vector
    

class CppPaperProblemRAmax(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.1/(3**2), theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping=None):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K, update_clipping=update_clipping)
        self.optimal_solution = None
        self.optimal_value = None
        self.objective_vector = np.array([1,1,1])
        self.constraint_matrix = np.array([[3,12,2],[10,3,5], [5,3,15]])
        self.theta_G = 0.1#/(3) # s^2
        tqdm.write(f"theta_G: {self.theta_G}")

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        return np.random.lognormal(0,0.5**2, (self.I, *self.dimension))

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return (z-np.dot(self.constraint_matrix, x)).max()

    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function w.r.t x"""
        return -self.constraint_matrix.max(axis=0)

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
        return np.dot(self.objective_vector, x)

    def partial_f_function(self, x) -> np.ndarray:
        return self.objective_vector