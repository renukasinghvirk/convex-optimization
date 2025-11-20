# %%%%%%%%%%%%%%%%%%%%%% MGT-418 Convex Optimization %%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------- Load data -------------------
mat = loadmat('p2data1.mat')
#mat = loadmat('p2data2.mat') #uncomment to solve for the second data set

x = np.asarray(mat['x'], dtype=float)
y = np.asarray(mat['y'], dtype=float).reshape(-1)
m, d = x.shape
# ------------------- Parameters -------------------
rho = 1e-4 # 0.5  # regularization parameter

# Solve SVM problem with smooth Hinge loss to compute the SVM coefficients 
# w and b (denote them by w and b and describe w as a column vector)
# +-----------------+
# define decision variables
w = cp.Variable((d,1))
b = cp.Variable(1)
t = cp.Variable(m)
s = cp.Variable(m)
# define the constraints
constraints = []
for i in range(m):
    constraints.append(0.5*cp.square(t[i]) + 1 - y[i]*(w.T @ x[i] - b) + t[i] <= s[i])
    constraints.append(0.5*cp.square(t[i]) <= s[i])

   
objective = cp.Minimize((1/m) * cp.sum(s) + (rho/2)*cp.sum_squares(w))
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)
w_val = w.value
b_val = b.value.item()
# +-----------------+

print(f"Objective*: {prob.value:.6f}  ||w||={np.linalg.norm(w_val):.4f}  b={b_val:.4f}")

# ------------------- Discretization & labels (100 points per feature) -------------------
# Discretize each feature range to 100 discretization points to get 100^d
# total number of discretization points in the feature space
# Construct a feature matrix (denote by feature) of discretization points
# Specifically, feature will be a matrix in R^((100^d) x d), where each row
# represents a distinct feature vector
# Compute the label of each discrete point by using optimal w and b
# Construct a label vector (denoted by label) containing the respective labels
# Specifically, label will be a vector in R^((100^d) x 1)

# +-----------------+
# Discretize each feature range to 100 discretization points to get 100^d
# total number of discretization points in the feature space

xmin = x.min(axis=0)
xmax = x.max(axis=0)
# generate 100 points per feature (d)
num_points = 100
discretization_points = [np.linspace(xmin[i], xmax[i], num_points) for i in range(d)] # dx100
assert(np.shape(discretization_points)==(d,num_points))

# Construct a feature matrix (denote by feature) of discretization points
feature = np.meshgrid(*discretization_points)
feature = np.vstack([m.flatten() for m in feature]).T
assert(np.shape(feature)==(num_points**d,d))

# Compute the label of each discrete point by using optimal w and b
# label = w^Tx-b
label = feature @ w_val - b_val
# Construct a label vector (denoted by label) containing the respective labels
label = label.flatten()
# Specifically, label will be a vector in R^((100^d) x 1)
assert(np.shape(label)==(num_points**d,))
# +-----------------+

# ------------------- Visualization -------------------
# feel free to comment out and construct your own plots.
m_light_red  = label >= 1
m_dark_red   = (label >= 0) & (label < 1)
m_dark_blue  = (label < 0)  & (label > -1)
m_light_blue = label <= -1
print("feature shape:", feature.shape)
print("label shape:", label.shape)
print("m_light_blue shape:", m_light_blue.shape)

plt.figure(figsize=(7, 6))
ax = plt.gca()
ax.set_facecolor("white")  # improve contrast
 
# plot light regions first (more transparent)
plt.scatter(feature[m_light_blue, 0], feature[m_light_blue, 1],
            s=23, c=[[0, 0.3, 1, 0.25]], marker='.', edgecolors='none',
            zorder=1, rasterized=True)
plt.scatter(feature[m_light_red, 0], feature[m_light_red, 1],
            s=23, c=[[1, 0.3, 0, 0.25]], marker='.', edgecolors='none',
            zorder=2, rasterized=True)
 
# plot dark regions on top (less transparent)
plt.scatter(feature[m_dark_blue, 0], feature[m_dark_blue, 1],
            s=23, c=[[0, 0, 1, 0.6]], marker='.', edgecolors='none',
            zorder=3, rasterized=True)
plt.scatter(feature[m_dark_red, 0], feature[m_dark_red, 1],
            s=23, c=[[1, 0, 0, 0.6]], marker='.', edgecolors='none',
            zorder=4, rasterized=True)
 
# training points on top
train_red  = (y >= 1)
train_blue = ~train_red
plt.scatter(x[train_blue, 0], x[train_blue, 1],
            s=35, c=[[0, 0, 1, 1.0]], marker='o', edgecolors='none',
            zorder=5)
plt.scatter(x[train_red, 0], x[train_red, 1],
            s=35, c=[[1, 0, 0, 1.0]], marker='o', edgecolors='none',
            zorder=6)
 
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('p2data1_rho_05.png', dpi=1200)
#plt.savefig('p2data1.png', dpi=1200)
#plt.savefig('p2data2.png', dpi=1200)
plt.show()
