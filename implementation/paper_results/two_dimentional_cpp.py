import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
sys.path.insert(0, ".")
import json, pickle
from problem_instances import problem_instances

ZOOM = [None, 1/35][-1]

XLIM = (2.5, 6)
YLIM = (2.5, 9)
# XLIM = (-3, 2)
# YLIM = (-3, -0.25)
# XLIM = (-3.5, 1.5)
# YLIM = (-1, 4)
MARKER_SIZE = 14

problem = "CppPaperProblem1_0"
pb = problem_instances[problem]
#TITLE = f"{problem} results from 100 runs"
INPUT_PATH = os.path.join(os.path.dirname(__file__),f"convergence_{problem}.pkl")
#OUTPUT_PATH = __file__.replace(".py", ".pdf")
OUTPUT_PATH = f"paper_results/plots/two_dimentional_cpp.pdf"

CPP_DIR = "paper_results/cpp/CppPaperProblem1_0/"
FILES = os.listdir(CPP_DIR)

INPUT_CPP = "paper_results/cpp/CppPaperProblem1_0/CPP-KKT_N300_K100_V1000_delta0.05.json"
cpp_path = np.array(json.load(open(INPUT_CPP, "r")))

def import_json(path):
    return json.load(open(path, "r"))

cpp_paths = [(import_json(os.path.join(CPP_DIR, f)), f.split("_")[0]) for f in FILES if not "old" in f and ("SAA" in f or "CPP" in f)]
print(cpp_paths)

if ZOOM:
    OUTPUT_PATH = OUTPUT_PATH.replace(".pdf", "_zoom.pdf")

def get_taco_results():
    dir_ = f"paper_results/taco_res/"
    files = os.listdir(dir_)
    file = [f for f in files if f.endswith(f"{problem}.json")][0]
    print(f"Found {len(files)} files in {dir_}")
    return np.array(import_json(os.path.join(dir_, file))["x_history"])


taco_path = get_taco_results()

def import_results(path):
    """import the results from a pickle file"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results


res_gradient, res_zeroth_order = import_results(INPUT_PATH)


# pb = CppPaperProblem1_0(
#        initial_x=[3,3], N=10, T=500, I=500, max_iters=100, lambda_=0.1, GA_steps=300, theta_G=1-0.95, theta_H=0.9,
#       learning_rate_SGLD=1e-1, learning_rate_GD=3e-2, learning_rate_GA=0.1, delta=1e-2, mu=2e0, K=3e-2, update_clipping=None
#     )

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


f_function = pb.f_function
chance_function = lambda x: pb.chance_function(x, z=15.5)


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
chance_threshold = -11
f_threshold = -18
# chance_threshold = 6
# f_threshold = 4
# chance_threshold = 8
# f_threshold = 12

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


#ax.plot(cpp_path[:, 0], cpp_path[:, 1], '-', color='green', alpha=1, linewidth=3)
#ax.plot(cpp_paths[1][0][:, 0], cpp_paths[1][0][:, 1], '-', alpha=1, linewidth=3)

ax.plot(gradient_path[:, 0], gradient_path[:, 1], '-', color='blue', alpha=1, linewidth=3)
# adding a black square at the start of the path
#ax.plot(0, 0 , marker='s', color='black', markersize=MARKER_SIZE, markeredgecolor='gold')
ax.plot(taco_path[:, 0], taco_path[:, 1], '-',color='green', alpha=1, linewidth=3)
taco_final_point = taco_path[-1]
# Final point with star
zero_order_final_point = zero_order_path[-1]
ax.plot(zero_order_final_point[0], zero_order_final_point[1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='purple')

#cpp_final_point = cpp_path[-1]
ax.plot(cpp_paths[0][0]["x_history"][-1][0], cpp_paths[0][0]["x_history"][-1][1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='red')
ax.plot(cpp_paths[1][0]["x_history"][-1][0], cpp_paths[1][0]["x_history"][-1][1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='fuchsia')
ax.plot(cpp_paths[2][0]["x_history"][-1][0], cpp_paths[2][0]["x_history"][-1][1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor="dodgerblue")
#ax.plot(cpp_paths[3][0]["x_history"][-1][0], cpp_paths[2][0]["x_history"][-1][1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor="grey")

gradient_final_point = gradient_path[-1]
ax.plot(gradient_final_point[0], gradient_final_point[1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='blue')
ax.plot(taco_final_point[0], taco_final_point[1], marker='^', color='gold', markersize=MARKER_SIZE, markeredgecolor='green')

# Labels, limits, legend
if ZOOM:
    mean_point = (zero_order_final_point + cpp_paths[0][0]["x_history"][-1] + gradient_final_point) / 3
    XLIM = (mean_point[0] - ZOOM, mean_point[0] + ZOOM)
    YLIM = (mean_point[1] - ZOOM, mean_point[1] + ZOOM)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

legend_elements = [
    Line2D([0], [0], color='red', lw=4, label='Chance function with $Z=\mu=15.5$ (filled)'),
    Line2D([0], [0], color='blue', lw=4, label='Objective function (filled)'),
    #Line2D([0], [0], color='gold', lw=2, linestyle='dashed', label='Chance = 0 contour'),
    #Line2D([0], [0], color='gold', lw=2, linestyle='dashed', label='f = 2.0 contour'),
    Line2D([0], [0], color='blue', lw=2, marker='_', label='Gradient based optimization trajectory'),
    #Line2D([0], [0], color='green', lw=2, marker='_', label='CPP-KKT optimization trajectory'),
    Line2D([0], [0], color='purple', lw=2, marker='_', label='Zeroth order optimization trajectory'),
    Line2D([0], [0], color='green', lw=2, marker='_', label='Bundle optimization trajectory'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='blue', lw=0, label='Gradient based final point'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='purple', lw=0, label='Zeroth order final point'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='green', lw=0, label='Bundle final point'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='red', lw=0, label='CPP-KKT algorithm final point'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='fuchsia', lw=0, label='CPP-MIP algorithm final point'),
    #Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='dodgerblue', lw=0, label='SA algorithm final point'),
    Line2D([0], [0], marker='^', color='gold', markersize=MARKER_SIZE*0.9, markeredgecolor='dodgerblue', lw=0, label='SAA algorithm final point')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2)

plt.tight_layout(rect=[0, 0.26, 1, 1])
plt.savefig(OUTPUT_PATH)
plt.show()
