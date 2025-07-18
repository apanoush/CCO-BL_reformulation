import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import bokeh.plotting as bp
from tqdm import tqdm
#from problem import Problem
#from algorithm import Algorithm

def plot_s(s_history: np.ndarray, max_alpha: np.ndarray=None, title=None):
    """plots the steps for the SGLD algorithm on the s space

    Args:
        s_history (np.ndarray): each vector in the matrix is a step in the SGLD algorithm; each row is a different s
        min_alpha (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.
    """
    
    plt.figure(figsize=(10, 5))
    for step in s_history:
        plt.scatter(range(len(step)), step, s=1, alpha=0.6)

    if max_alpha:
        plt.scatter(s_history.shape[1], max_alpha, s=10, label="min_alpha")

    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    if title: plt.title(title)
    else: plt.title('1D point distribution over SGLD Steps')
    plt.show()

def plot_alpha(alpha_history: np.ndarray, final_s, title='1D point distribution over GA Steps', figsize=(10,5)):
    """plots the steps for the GA algorithm on the s space
        Args:
        alpha_history (np.ndarray): row vector of the alpha values
        final_s (np.ndarray): the final s distribution
    """
    
    plt.figure(figsize=figsize)

    plt.scatter(range(alpha_history.shape[0]), alpha_history, c='blue', marker='x', s=10, alpha=0.6, label="alpha")

    plt.scatter(np.zeros_like(final_s), final_s, s=10, c='red', marker='o', label="final s distribution")

    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()

def plot_final_s(final_s, title="last iteration of SGLD on the s distribution", figsize=(10, 5), xlabel = 'Value', max_alpha=None):
    plt.figure(figsize=figsize) 
    plt.scatter(final_s, np.zeros_like(final_s), alpha=0.5, c='red', marker='o', label="final s distribution")
    if max_alpha: 
        plt.scatter(max_alpha, 0, c='blue', marker='x', label="max alpha")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    # hide the y-axis
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.ticklabel_format(useOffset=False, style='plain')
    plt.grid(axis='x')

    plt.tight_layout()
    plt.show()


def plot_gd(x_history, f_history, title="GD", figsize=(12, 5)):
    plt.figure(figsize=figsize)

    # scatter plot of all 2D x values
    plt.scatter(*zip(*x_history), c=f_history, cmap='viridis', s=20)
    plt.colorbar()

    plt.title(title)
    plt.show()

def plot_results_1D(algorithm, problem, title=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # scatter plot of all 2D x values
    ax1.scatter(algorithm.x_history, algorithm.objective_function_history, s=20)

    ax1.set_title("H function w.r.t. x")
    ax1.set_ylabel("H function")
    ax1.set_xlabel("x")
    ax1.grid(True)
    
    # scatter plot of all 2D x values, red if not feasible and green if feasible, cmap should be binary
    #cmap = matplotlib.colors.ListedColormap(['red', 'green'])
    ax2.scatter(algorithm.x_history, algorithm.chance_constraint_quantile_history, s=20) 
    ax2.set_ylabel("Feasibility of x / chance_constraint_quantile")
    ax2.set_xlabel("x")
    ax2.grid(True)

    ax2.set_title("Feasibility of x (chance_constraint_quantile)")
    #ax2.set_aspect("equal", )
    if not title:
        fig.suptitle(f"Results of our {problem.max_iters} steps optimization algorithm with initial x={problem.initial_x}")
    else: fig.suptitle(title)
    fig.tight_layout()

    plt.show()


def plot_results_2D(algorithm, problem, title = None, log = False):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ("x_1", "x_2")
    scatter_norm = "linear"
    title_log = ""

    if log:
        scatter_norm = "log"
        title_log = "Log "

    # scatter plot of all 2D x values
    ax1.scatter(*zip(*algorithm.x_history), c=algorithm.objective_function_history, cmap='viridis', s=20, norm=scatter_norm)
    fig.colorbar(mappable=ax1.collections[0], ax=ax1)
    # show the axis
    # plt.axhline(0, color='black', lw=1)
    # plt.axvline(0, color='black', lw=1)
    ax1.set_title(title_log + "H function w.r.t. x")
    #ax1.set_aspect("equal")
    ax1.grid(True) 

    if algorithm.chance_constraint_quantile_history is None:
        plt.show()
        return
    
    # scatter plot of all 2D x values, red if not feasible and green if feasible, cmap should be binary
    #cmap = matplotlib.colors.ListedColormap(['red', 'green'])
    ax2.scatter(*zip(*algorithm.x_history), c=algorithm.chance_constraint_quantile_history, s=20, norm=scatter_norm) 
    ax2.grid(True)
    fig.colorbar(mappable=ax2.collections[0], ax=ax2)
    # show the axis
    # plt.axhline(0, color='black', lw=1)
    # plt.axvline(0, color='black', lw=1)

    ax2.set_title(title_log + "Feasibility of x / chance_constraint_quantile")

    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])

    #ax2.set_aspect("equal", )
    if not title:
        fig.suptitle(f"Results of our {problem.max_iters} steps optimization algorithm with initial x={problem.initial_x}")
    else: fig.suptitle(title)
    fig.tight_layout()

    plt.show()


def plot_suboptimality(algorithm, savepath=None, figsize=(12, 4)):
    """(f(x_k)−f^*)/|f^*|
    """
    assert algorithm.empirical_coverage, "Suboptimality plot is only available when empirical coverage is True"
    assert algorithm.problem.optimal_value is not None, "Optimal value is not available"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    suboptimality = algorithm.oracle.suboptimality_f(algorithm)
    iterations = np.arange(len(suboptimality))
    ax1.plot(iterations, suboptimality, label="Suboptimality")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Sub-optimality")
    ax1.set_title("Sub-optimality of the optimal value")
    ax1.grid(True)

    empirical_coverage_suboptimality = algorithm.oracle.supoptimality_ec(algorithm)
    ax2.plot(iterations, empirical_coverage_suboptimality, label="Empirical coverage")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Sup-optimiality")
    ax2.set_title("Sup-optimiality of the empirical coverage")
    ax2.grid(True)

    fig.tight_layout()

    if savepath:
        try:
            plt.savefig(savepath, dpi=300)
        except:
            print(f"Could not save the plot to {savepath}")

    plt.show()


def plot_z_samples(z_samples, title=None):
    plt.scatter(*zip(*z_samples), c='blue')
    if title: plt.title(title)
    plt.show()


def plot_1D_func(x, y= None, func = None, title=None, figsize=(10, 5), xlabel=None, ylabel=None, verbose=False):
    
    # checking that both y and func are not None
    assert not (y is None and func is None), "Both y and func cannot be None"

    plt.figure(figsize=figsize)

    if not y and func:
        y = [func(x) for x in tqdm(x, desc="Computing f function", disable=not verbose)]

    plt.plot(x, y)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_multiple_1D_func(x, y: np.ndarray, title=None, figsize=(10, 5), xlabel=None, ylabel=None, labels=None):
    """_summary_

    Args:
        x (_type_): _description_
        y (np.ndarray): each row is a function to plot
        title (_type_, optional): _description_. Defaults to None.
        figsize (tuple, optional): _description_. Defaults to (10, 5).
        xlabel (_type_, optional): _description_. Defaults to None.
        ylabel (_type_, optional): _description_. Defaults to None.
    """

    plt.figure(figsize=figsize)

    if labels:
        assert len(labels) == y.shape[0], "Number of labels must be equal to number of functions"

    for i, y_ in enumerate(y):
        if not labels: plt.plot(x, y_)
        else: plt.plot(x, y_, label=labels[i])

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if labels: plt.legend() 
    plt.grid(True)
    plt.show()

def plot_2D_func(x, y=None, func=None, title=None, figsize=(10, 5), xlabel= "x_1", ylabel="x_2"):
        
    # checking that both y and func are not None
    assert not (y is None and func is None), "Both y and func cannot be None"

    if not y and func:
        y = [func(x) for x in x]

    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.scatter(*zip(*x), c=y, cmap='viridis', s=20)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()

def plot_2D_func_vectors(x, func, title=None):
    y = [func(x) for x in x]
    plt.quiver(*zip(*x), *zip(*y))#, angles='xy', scale_units='xy', scale=1)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()

def plot_2D_func_both(x, func, title=None, verbose=False):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    y = [func(x) for x in tqdm(x, desc="Computing f function", disable=not verbose)]
    y_mean = np.mean(y, axis=1)

    ax1.scatter(*zip(*x), c=y_mean, cmap='viridis', s=20)
    fig.colorbar(mappable=ax1.collections[0], ax=ax1)
    ax1.set_aspect("equal")
    ax1.set_title("f function mean output")
    ax1.grid(True)

    ax2.quiver(*zip(*x), *zip(*y), color="blue") #streamplot
    #plt.quiverkey(ax2, 0.9, 0.9, 1, 'quiver key', labelpos='E')
    ax2.set_aspect("equal")
    ax2.set_title("f function vectors output")
    ax2.grid(True)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()

def plot_2D_func_both_and_gd(x, func, x_history, f_history, title=None, verbose=False):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    y = [func(x) for x in tqdm(x, desc="Computing f function", disable=not verbose)]
    y_mean = np.mean(y, axis=1)

    ax1.scatter(*zip(*x), c=y_mean, cmap='viridis', s=20)
    fig.colorbar(mappable=ax1.collections[0], ax=ax1)
    ax1.set_aspect("equal")
    ax1.set_title("f function mean output")
    ax1.grid(True)

    ax2.quiver(*zip(*x), *zip(*y), color="blue") #streamplot
    #plt.quiverkey(ax2, 0.9, 0.9, 1, 'quiver key', labelpos='E')
    ax2.set_aspect("equal")
    ax2.set_title("f function vectors output")
    ax2.grid(True)

    if title:
        fig.suptitle(title)

    # scatter plot of all 2D x values
    ax2.scatter(*zip(*x_history), c=f_history, cmap="viridis", s=10)
    fig.colorbar(mappable=ax2.collections[0], ax=ax2)

    fig.tight_layout()
    plt.show()


def plot_2D_func_bokeh(x, func, title=None):
    y = [func(x) for x in x]
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,examine,help"
    p = bp.figure(tools=TOOLS)
    p.scatter(*zip(*x), color=y)
    if title:
        p.title.text = title
    
    bp.output_file("2D_func.html")
    bp.show(p)

    