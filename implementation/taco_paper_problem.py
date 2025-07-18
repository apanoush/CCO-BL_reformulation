from problem import Problem
import numpy as np
from scipy import stats

"""
best hyperparameters values:
- theta_G: 0.1
- theta_H: 0.1
- learning_rate_SGLD: 0.5?, 5? 40?
- learning_rate_GA: 0.01 ? 20000
- learning_rate_GD: 0.001
- lambda: 0.1, 0.001, 1e-5
- I: 500, 600
- T: 500? 200
- GA_steps=500

initial_x=[2,1], N=50, T=500, I=600, GA_steps=2200, max_iters=20, theta_G=0.1, theta_H=100, lambda_=0.001, learning_rate_SGLD=40, learning_rate_GA=5000, learning_rate_GD=10, mu=100, delta= 1000
initial_x=[2,1], N=10, T=500, I=600, GA_steps=2200, max_iters=20, theta_G=0.1, theta_H=0.8, lambda_=0.01, learning_rate_SGLD=40, learning_rate_GA=80, learning_rate_GD=0.1, mu=100, delta= 1000

initial_x=[2,1], N=10, T=500, I=400, GA_steps=1500, max_iters=20, theta_G=0.1, theta_H=0.85, lambda_=0.01, learning_rate_SGLD=40, learning_rate_GA=100, learning_rate_GD=0.05, mu=200, delta= 3000   - no full x-partial H

initial_x=[2,1], N=10, T=300, I=400, GA_steps=300, max_iters=20, theta_G=0.2, theta_H=0.85, lambda_=0.01, learning_rate_SGLD=40, learning_rate_GA=200, learning_rate_GD=0.01, mu=100, delta= 50

actually working:
initial_x=[2,1], N=12, T=300, I=300, GA_steps=300, max_iters=50, theta_G=0.03368421, theta_H=1-0.03368421, lambda_=0.01, learning_rate_SGLD=60, learning_rate_GA=2000, learning_rate_GD=0.01, mu=100, delta= 50

initial_x=[2,1], N=12, T=220, I=300, GA_steps=300, max_iters=100, theta_G=0.03368421, theta_H=0.9999, lambda_=0.01, learning_rate_SGLD=60, learning_rate_GA=2000, learning_rate_GD=0.003, mu=10, delta= 5

initial_x=[2,1], N=12, T=220, I=300, GA_steps=300, max_iters=250, theta_G=0.03368421, theta_H=0.9999, lambda_=0.01, learning_rate_SGLD=60, learning_rate_GA=500, learning_rate_GD=0.003, mu=1000, delta= 5

initial_x=[2,1], N=40, T=100, I=300, GA_steps=330, max_iters=250, theta_G=0.03368421, theta_H=0.9999, lambda_=0.01, learning_rate_SGLD=210, learning_rate_GA=200000, learning_rate_GD=0.025, mu=1, delta= 5


problem = TacoPaperProblem1(
       initial_x=[2,1], N=10, T=1000, I=400, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e2, learning_rate_GA=7e4, learning_rate_GD=1e-2, mu=4e-2, delta= 5e-2, K=1e-5, update_clipping=1/2
    )
initial_x=[2,1], N=10, T=1000, I=400, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e2, learning_rate_GA=7e4, learning_rate_GD=5e-2, mu=8e-3, delta= 1e-2, K=1e-5, update_clipping=1/2
initial_x=[2,1], N=10, T=1000, I=400, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e2, learning_rate_GA=7e4, learning_rate_GD=8e-2, mu=1e-4, delta= 1e-2, K=1e-5, update_clipping=1/2
mean:0
initial_x=[2,1], N=10, T=1000, I=400, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e2, learning_rate_GA=7e4, learning_rate_GD=8e-2, mu=1e-6, delta= 1e-2, K=1e-5, update_clipping=1/2
"""


class TacoPaperProblem1(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.1, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping:float=None):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K, update_clipping)

        self.a = np.array([2.0, 2.0], dtype=np.float64)
        r = np.sqrt(2)/2
        rot = np.array([[r, -1.0 * r], [r, r]], dtype=np.float64)
        inv_rot = np.array([[r, r], [-1.0 * r, r]], dtype=np.float64)
        mat_in = np.array([[1., 0.], [0., 10.]], dtype=np.float64)
        self.geo_a = np.dot(inv_rot, np.dot(mat_in, rot))

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        #print(np.random.normal([1, 1], [20, 20] , (self.I, self.dimension)).shape)
        #return np.random.randn(self.I, self.dimension)
        np.random.seed(42)
        dimension = self.dimension[0]
        mean = np.array([1]*dimension)
        cov = 20 * np.eye(dimension)
        nb_samples=self.I
        data = np.random.multivariate_normal(mean, cov, size=nb_samples)
        return np.asarray(data, dtype=np.float64)

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        beta = np.array([1., 1.], dtype=np.float64)
        a = np.dot(np.transpose(z), np.dot(self.mat_w(x), z))
        b = np.dot(beta, z)
        c = -1.
        res = a + b + c
        return res
    
    # def chance_function(self, x, z) -> float:

    #     W = np.zeros((2,2))
    #     W[0,0] = x[0]**2 + 0.5
    #     W[1,1] = abs((x[1] - 1))**3 + 0.2

    #     return np.dot(np.dot(z, W), z) + np.dot(np.array([1, 1]), z) -1

    #@staticmethod
    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function"""
        g0 = 2 * x[0] * z[0]**2
        g1 = 3 * np.sign(x[1] - 1.0) * (z[1] * (x[1] - 1.0))**2
        res = np.array([g0, g1])
        return res

    def h_function(self, x: np.ndarray) -> np.ndarray:
        """relu function"""

        # if self.dimension == 1 or type(x) == np.float64:
        #     return np.maximum(x, 0)
        # else:
        #     return np.array([np.maximum(e, 0) for e in x])

        #return np.maximum(x, 0)

        #x = np.array(x)

        #return (-1/(16*delta**3))*x**4 + (3/(8*delta))*x**2 + (1/2)*x + (3/16)*delta

        #return (1/2)*(x + np.sqrt(x**2 + 4*delta**2))


        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, x, (-1/(16*self.delta**3))*x**4 + (3/(8*self.delta))*x**2 + (1/2)*x + (3/16)*self.delta)
                        )

        conditions = [
            x <= -self.delta,        
            x >= self.delta           
        ]
        functions = [
            0,                   
            x,                   
            (-1/(16*self.delta**3))*x**4 + (3/(8*self.delta))*x**2 + (1/2)*x + (3/16)*self.delta
        ]
        return np.piecewise(x, conditions, functions)

    def partial_h_function(self, x: np.ndarray) -> np.ndarray:
        """differentiation of the h function"""
        # differentiation of the ReLU function

        # if self.dimension == 1 or type(x) == np.float64:
        #     return np.where(x > 0, 1, 0)
        # else:
        #     return np.array([np.where(e > 0, 1, 0) for e in x])

        #x = np.array(x)

        #return (1/2)*(1 + x/np.sqrt(x**2 + 4*delta**2))

        #return np.where(x > 0, 1, 0)

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, 1, (-1/(4*self.delta**3))*x**3 + (3/(4*self.delta))*x + (1/2))
            )

        conditions = [
            x <= -self.delta,        
            x >= self.delta           
        ]
        functions = [
            0,                   
            1,                   
            (-1/(4*self.delta**3))*x**3 + (3/(4*self.delta))*x + (1/2)
        ]
        return np.piecewise(x, conditions, functions)

    def f_function(self, x):
        return 0.5 * np.dot(x - self.a, np.dot(np.ascontiguousarray(self.geo_a), x - self.a))

    def partial_f_function(self, x) -> np.ndarray:
        return np.dot(self.geo_a, x - self.a)
    
    @staticmethod
    def mat_w(x):
        d1 = x[0] ** 2 + 0.5
        d2 = abs((x[1] - 1)) ** 3 + 0.2
        res = np.diag(np.array([d1, d2], dtype=np.float64))
        return res
    

class TacoPaperProblem1_2(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.1, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping:float=None):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K, update_clipping)

        self.a = np.array([-2.0, -3.0], dtype=np.float64)
        r = np.sqrt(2)/2
        rot = np.array([[r, -1.0 * r], [r, r]], dtype=np.float64)
        inv_rot = np.array([[r, r], [-1.0 * r, r]], dtype=np.float64)
        mat_in = np.array([[2., 0.], [0., 4.]], dtype=np.float64)
        self.geo_a = np.dot(inv_rot, np.dot(mat_in, rot))

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        #print(np.random.normal([1, 1], [20, 20] , (self.I, self.dimension)).shape)
        #return np.random.randn(self.I, self.dimension)
        np.random.seed(42)
        dimension = self.dimension[0]
        mean = np.array([1]*dimension)
        cov = 20 * np.eye(dimension)
        nb_samples=self.I
        data = np.random.multivariate_normal(mean, cov, size=nb_samples)
        return np.asarray(data, dtype=np.float64)

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        beta = np.array([1., 1.], dtype=np.float64)
        a = np.dot(np.transpose(z), np.dot(self.mat_w(x), z))
        b = np.dot(beta, z)
        c = -1.
        res = a + b + c
        return res
    
    # def chance_function(self, x, z) -> float:

    #     W = np.zeros((2,2))
    #     W[0,0] = x[0]**2 + 0.5
    #     W[1,1] = abs((x[1] - 1))**3 + 0.2

    #     return np.dot(np.dot(z, W), z) + np.dot(np.array([1, 1]), z) -1


    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function"""
        g0 = 2 * (x[0] -3 ) * z[0]**2
        g1 = 3 * np.sign(x[1]+1) * (z[1] * (x[1]+1))**2
        res = np.array([g0, g1])
        return res

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
        return 0.5 * np.dot(x - self.a, np.dot(np.ascontiguousarray(self.geo_a), x - self.a))

    def partial_f_function(self, x) -> np.ndarray:
        return np.dot(self.geo_a, x - self.a)
    
    @staticmethod
    def mat_w(x):
        d1 = (x[0] - 3 ) ** 2 + 1
        d2 = abs(x[1] +1) ** 3 - 0.4
        res = np.diag(np.array([d1, d2], dtype=np.float64))
        return res
    

class TacoPaperProblem1_3(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.1, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1, update_clipping:float=None):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K, update_clipping)

        self.a = np.array([2.0, -3.0], dtype=np.float64)
        r = np.sqrt(2)/2
        rot = np.array([[r, -1.0 * r], [r, r]], dtype=np.float64)
        inv_rot = np.array([[r, r], [-1.0 * r, r]], dtype=np.float64)
        mat_in = np.array([[1., 0.], [0., 5.]], dtype=np.float64)
        self.geo_a = np.dot(inv_rot, np.dot(mat_in, rot))

    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        #print(np.random.normal([1, 1], [20, 20] , (self.I, self.dimension)).shape)
        #return np.random.randn(self.I, self.dimension)
        np.random.seed(42)
        dimension = self.dimension[0]
        mean = np.array([1]*dimension)
        cov = 20 * np.eye(dimension)
        nb_samples=self.I
        data = np.random.multivariate_normal(mean, cov, size=nb_samples)
        return np.asarray(data, dtype=np.float64)

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        beta = np.array([1., 1.], dtype=np.float64)
        a = np.dot(np.transpose(z), np.dot(self.mat_w(x), z))
        b = np.dot(beta, z)
        c = -1.
        res = a + b + c
        return res
    
    # def chance_function(self, x, z) -> float:

    #     W = np.zeros((2,2))
    #     W[0,0] = x[0]**2 + 0.5
    #     W[1,1] = abs((x[1] - 1))**3 + 0.2

    #     return np.dot(np.dot(z, W), z) + np.dot(np.array([1, 1]), z) -1


    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function"""
        g0 = 2 * (x[0] + 2 ) * z[0]**2
        g1 = 3 * np.sign(x[1]- 3) * (z[1] * (x[1]-3))**2
        res = np.array([g0, g1])
        return res

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
        return 0.5 * np.dot(x - self.a, np.dot(np.ascontiguousarray(self.geo_a), x - self.a))

    def partial_f_function(self, x) -> np.ndarray:
        return np.dot(self.geo_a, x - self.a)
    
    @staticmethod
    def mat_w(x):
        d1 = (x[0] + 2 ) ** 2 
        d2 = abs(x[1] - 3) ** 3 
        res = np.diag(np.array([d1, d2], dtype=np.float64))
        return res
    
    
class TacoPaperProblem2(Problem):
    """PaperProblem class, containing
    all the useful parameters (e.g. N, T, I, theta, lambda, learning_rate, h_function, and chance_function)"""

    def __init__(self, initial_x, N, T, I, max_iters, GA_steps=15, unique_chance_constraint_minimizer=True, theta_G=0.1, theta_H=1, lambda_=0.1, learning_rate_SGLD=0.1, learning_rate_GD=0.1, learning_rate_GA=0.1, mu=1, delta=0.5, initial_alpha=1, K=1):
        super().__init__(initial_x, N, T, I, max_iters, GA_steps, unique_chance_constraint_minimizer, theta_G, theta_H, lambda_, learning_rate_SGLD, learning_rate_GD, learning_rate_GA, mu, delta, initial_alpha, K)
        self.optimal_value = - (10 * self.dimension[0]) / np.sqrt(stats.chi2.ppf((1-self.theta_G)**(1/10), df=self.dimension[0]))


    def z_samples(self):
        """generate samples for z"""
        # returns a vector of I samples of self.dimension from a parametrized random distribution
        return np.random.normal(0 , 1, (self.I, 10, *self.dimension))
        #return np.random.randn(self.I, self.dimension)
        # np.random.seed(42)
        # dimension = self.dimension
        # mean = np.array([0.0]*dimension)
        # cov = 20 * np.eye(dimension)
        # nb_samples=10000
        # data = np.random.multivariate_normal(mean, cov, size=nb_samples)
        # return np.asarray(data, dtype=np.float64)

    def chance_function(self, x, z) -> float:
        """chance function, is not deterministic"""
        return np.max(np.dot(np.array(z)**2 , np.array(x)**2)) - 100

    #@staticmethod
    def partial_chance_function(self, x, z) -> np.ndarray: 
        """differentiation of the chance function"""
        return 2 * np.max(np.dot(np.array(z)**2 , np.array(x)))

    def h_function(self, x: np.ndarray) -> np.ndarray:
        """relu function"""

        # if self.dimension == 1 or type(x) == np.float64:
        #     return np.maximum(x, 0)
        # else:
        #     return np.array([np.maximum(e, 0) for e in x])

        #return np.maximum(x, 0)

        #x = np.array(x)

        #return (-1/(16*delta**3))*x**4 + (3/(8*delta))*x**2 + (1/2)*x + (3/16)*delta

        #return (1/2)*(x + np.sqrt(x**2 + 4*delta**2))


        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, x, (-1/(16*self.delta**3))*x**4 + (3/(8*self.delta))*x**2 + (1/2)*x + (3/16)*self.delta)
                        )

    def partial_h_function(self, x: np.ndarray) -> np.ndarray:
        """differentiation of the h function"""
        # differentiation of the ReLU function

        # if self.dimension == 1 or type(x) == np.float64:
        #     return np.where(x > 0, 1, 0)
        # else:
        #     return np.array([np.where(e > 0, 1, 0) for e in x])

        #x = np.array(x)

        #return (1/2)*(1 + x/np.sqrt(x**2 + 4*delta**2))

        #return np.where(x > 0, 1, 0)

        return np.where(x <= -self.delta, 0, 
                        np.where(x >= self.delta, 1, (-1/(4*self.delta**3))*x**3 + (3/(4*self.delta))*x + (1/2))
            )
    def f_function(self, x, epsilon=1e-2):
        """-||x||_1"""
        #return -np.sum(np.abs(x))
        # we use a smooth approximation of the l1 norm 
        return -np.sum(np.sqrt(np.array(x)**2 + epsilon)-np.sqrt(epsilon))

    def partial_f_function(self, x, epsilon=1e-2) -> np.ndarray:
        #return -np.sign(x)
        # we use a smooth approximation of the l1 norm
        return -np.array(x) / np.sqrt(np.array(x)**2 + epsilon)
    
