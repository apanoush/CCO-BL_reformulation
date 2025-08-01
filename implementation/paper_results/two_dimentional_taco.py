import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
sys.path.insert(0, ".")
from taco_paper_problem import TacoPaperProblem1, TacoPaperProblem1_2, TacoPaperProblem1_3
import pickle, json
from problem_instances import problem_instances

# if not None, zooms with value around mean of final points
ZOOM = [None, 1/2][0] # 1/2 for 2, 1 else
MARKER_SIZE = 14

problem = ["TacoPaperProblem1", "TacoPaperProblem1_2", "TacoPaperProblem1_3"][0]
pb = problem_instances[problem]
#TITLE = f"{problem} results from 100 runs"
INPUT_PATH = os.path.join(os.path.dirname(__file__),f"convergence_{problem}.pkl")
OUTPUT_PATH = f"paper_results/plots/two_dimentional_taco{problem[-1]}.pdf"
if ZOOM:
    OUTPUT_PATH = OUTPUT_PATH.replace(".pdf", "_zoom.pdf")

# INPUT_TACO = f"paper_results/taco_res/output_ToyProblem{problem[-1]}.pkl"
# taco_path = pickle.load(open(INPUT_TACO, "rb"))["x_history"]

def import_json(path):
    return json.load(open(path, "r"))
def get_taco_results():
    dir_ = f"paper_results/taco_res/"
    files = os.listdir(dir_)
    file = [f for f in files if f.endswith(f"{problem}.json")][0]
    print(f"Found {len(files)} files in {dir_}")
    return np.array(import_json(os.path.join(dir_, file))["x_history"])

taco_path = get_taco_results()

XLIM = {
    "TacoPaperProblem1": (-1.25, 3.25),
    "TacoPaperProblem1_2": (-3, 2),
    "TacoPaperProblem1_3": (-3.5, 1.5)
}[problem]
YLIM = {
    "TacoPaperProblem1": (-0.5, 3),
    "TacoPaperProblem1_2": (-3, -0.25),
    "TacoPaperProblem1_3": (-1, 4)
}[problem]


def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results


res_gradient, res_zeroth_order = import_results(INPUT_PATH)

# pb = TacoPaperProblem1(
#     initial_x=[0, 0], N=10, T=2000, I=2000, GA_steps=500, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=1e0, learning_rate_GA=7e4, learning_rate_GD=2e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2
# )

# pb = TacoPaperProblem1_2(
#     initial_x=[0.39221721, -1.61773292], N=10, T=600, I=400, GA_steps=500, max_iters=400,
#     theta_G=1 - 0.03368421, theta_H=0.9, lambda_=0.01,
#     learning_rate_SGLD=1e2, learning_rate_GA=7e4, learning_rate_GD=5e-6,
#     mu=5e-1, delta=8e-1, K=1e-5
# )
# pb = TacoPaperProblem1_2(
#     initial_x=[1, -1], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=2e-1, update_clipping=1/2
# )
# pb = TacoPaperProblem1_3(
#     initial_x=[-2, 3], N=10, T=2000, I=500, GA_steps=180, max_iters=400, theta_G=1-0.03368421, theta_H=0.9, lambda_=0.01, learning_rate_SGLD=3e0, learning_rate_GA=7e4, learning_rate_GD=1.5e-1, mu=1e-3, delta= 1e-2, K=1.5e-1, update_clipping=1/2
# )

print(pb.geo_a)

f_function = pb.f_function
chance_function = lambda x: pb.chance_function(x, z=[1., 1.])


# --- Grid Setup ---
multipl_by_2 = lambda x: tuple(map(lambda y: y*2 , x))
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)

Z_chance = np.zeros_like(X)
Z_f = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z_chance[i, j] = chance_function(point)
        Z_f[i, j] = f_function(point)

# Thresholds
chance_threshold = {
    "TacoPaperProblem1": 4,
    "TacoPaperProblem1_2": 6,
    "TacoPaperProblem1_3": 8
}[problem]

f_threshold = {
    "TacoPaperProblem1": 2,
    "TacoPaperProblem1_2": 4,
    "TacoPaperProblem1_3": 12
}[problem]
# chance_threshold = 
# f_threshold = 
# chance_threshold = 
# f_threshold = 

Z_chance_masked = np.ma.masked_where(Z_chance > chance_threshold, Z_chance)
Z_f_masked = np.ma.masked_where(Z_f > f_threshold, Z_f)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.axis('equal')

# Filled contours
ax.contourf(X, Y, Z_chance_masked, levels=20, cmap='Reds', alpha=0.6)
ax.contourf(X, Y, Z_f_masked, levels=20, cmap='Blues', alpha=0.5)

# Contour lines with labels
cont_chance = ax.contour(X, Y, Z_chance_masked, levels=20, colors='red', linewidths=0.8)
cont_f = ax.contour(X, Y, Z_f_masked, levels=20, colors='blue', linewidths=0.8)
ax.clabel(cont_chance, inline=True, fontsize=8, fmt="%.1f")
ax.clabel(cont_f, inline=True, fontsize=8, fmt="%.1f")

# Dashed contours at specific levels
#chance_zero = ax.contour(X, Y, Z_chance, levels=[0], colors='gold', linestyles='dashed', linewidths=2)
#ax.clabel(chance_zero, inline=True, fontsize=10, fmt="Chance=0")

#f_thresh = ax.contour(X, Y, Z_f, levels=[f_threshold], colors='gold', linestyles='dashed', linewidths=2)
#ax.clabel(f_thresh, inline=True, fontsize=10, fmt="f=2.0")

zero_order_path = np.array(res_zeroth_order["x_history"])

ax.plot(zero_order_path[:, 0], zero_order_path[:, 1], '-', color='purple', alpha=1, linewidth=3)
gradient_path = np.array(res_gradient["x_history"])
# removing last iteration
gradient_path = gradient_path[:-1]


ax.plot(taco_path[:, 0], taco_path[:, 1], '-', color='green', alpha=1, linewidth=3)
ax.plot(gradient_path[:, 0], gradient_path[:, 1], '-', color='blue', alpha=1, linewidth=3)
# adding a black square at the start of the path
#ax.plot(0, 0 , marker='s', color='black', markersize=MARKER_SIZE, markeredgecolor='gold')

# Final point with star
zero_order_final_point = zero_order_path[-1]
ax.plot(zero_order_final_point[0], zero_order_final_point[1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='purple')

taco_final_point = taco_path[-1]
ax.plot(taco_final_point[0], taco_final_point[1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='green')

gradient_final_point = gradient_path[-1]
ax.plot(gradient_final_point[0], gradient_final_point[1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='blue')

# Labels, limits, legend
if ZOOM:
    mean_point = (zero_order_final_point + taco_final_point + gradient_final_point) / 3
    XLIM = (mean_point[0] - ZOOM, mean_point[0] + ZOOM)
    YLIM = (mean_point[1] - ZOOM, mean_point[1] + ZOOM)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

legend_elements = [
    Line2D([0], [0], color='red', lw=4, label='Chance function with $Z=\mu=[1,1]$ (filled)'),
    Line2D([0], [0], color='blue', lw=4, label='Objective function (filled)'),
    #Line2D([0], [0], color='gold', lw=2, linestyle='dashed', label='Chance = 0 contour'),
    #Line2D([0], [0], color='gold', lw=2, linestyle='dashed', label='f = 2.0 contour'),
    Line2D([0], [0], color='blue', lw=2, marker='_', label='Algorithm 1 optimization trajectory'),
    Line2D([0], [0], color='purple', lw=2, marker='_', label='Algorithm 2 optimization trajectory'),
    Line2D([0], [0], color='green', lw=2, marker='_', label='Bundle algorithm optimization trajectory'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='blue', lw=0, label='Algorithm 1 final point'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='purple', lw=0, label='Algorithm 2 final point'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='green', lw=0, label='Bundle algorithm final point')

]
fig.legend(handles=legend_elements, loc='lower center', ncol=2)

plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig(OUTPUT_PATH)
plt.show()
