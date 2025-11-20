# %%%%%%%%%%%%%%%%%%%%%% MGT-418 Convex Optimization %%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------- Load data -------------------
mat = loadmat("p2data2.mat")
x = np.asarray(mat['x'], dtype=float)
y = np.asarray(mat['y'], dtype=float).reshape(-1)
m, d = x.shape

# ------------------- Parameters -------------------
rho = 1e-4   # regularization parameter
sigma = 3.0  # bandwidth of Gaussian kernel

# Dual problem with Gaussian kernel
# Solve the dual problem (4) with the Gaussian kernel 
# Denote the dual decision variables by lambda
# +-----------------+
# define dual decision variable
Lambda = cp.Variable(m)

# define the constraints
constraints = [Lambda >= 0,
               Lambda <= 1/m,
               cp.sum(cp.multiply(Lambda, y)) == 0]


# gaussian kernel
def K(x, xp, sigma):
    sq_l2 = np.linalg.norm(x - xp)**2
    return np.exp(-sq_l2/(2*sigma**2))

# compute Φ(xi)^T Φ(xi') as well as double sum term of objective 
Kmat = np.zeros((m, m))
#double_sum = 0 -> leads to non DCP
for i in range(m):
    for j in range(m):
        Kmat[i, j] = K(x[i], x[j], sigma)
        #double_sum += Lambda[i] * Lambda[j] * y[i] * y[j] * Kmat[i, j] -> didn't work (not DCP)

M = np.diag(y) @ Kmat @ np.diag(y)
#double_sum = cp.sum(cp.multiply(Lambda, M @ Lambda)) -> not DCP

M_psd = cp.psd_wrap(M)
objective = cp.Maximize(
    cp.sum(Lambda - (m/2)*cp.square(Lambda))
    - (1/(2*rho)) * cp.quad_form(Lambda, M_psd)
)
#objective = cp.Maximize(cp.sum(Lambda - (m/2) * cp.square(Lambda)) - (1/(2*rho)) * double_sum) -> not DCP
prob = cp.Problem(objective)
prob.solve(solver=cp.MOSEK)
lambda_val = Lambda.value
# +-----------------+


# Compute optimal b (denote by b_opt) using the optimal dual solution

# +-----------------+
# define the bounds
upper = 1/m
lower = 1e-4

idx = np.where((lambda_val > lower) & (lambda_val < upper))[0]
k = idx[0]

yk = y[k]
lambda_k = lambda_val[k]

b_opt = yk * (m * lambda_k - 1) + (1/rho) * ((lambda_val * y) @ Kmat[:, k])
# +-----------------+

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
# similar as for exercise 2
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

# Compute the label of each discrete point by using optimal lambda and b
# label = 1/ρ Σ_{i=1}^m λ_i y_i K(x_i,x)-b
label = np.zeros(feature.shape[0])

for j in range(feature.shape[0]):
    x_new = feature[j]
    
    # compute (1/rho) * sum_i lambda_i y_i K(x_i, x_new)
    s = 0.0
    for i in range(m):
        s += lambda_val[i] * y[i] * K(x[i], x_new, sigma)
    label[j] = (1/rho) * s  -  b_opt

# Construct a label vector (denoted by label) containing the respective labels
label = label.flatten()
# Specifically, label will be a vector in R^((100^d) x 1)

assert label.shape == (num_points**d,)
# +-----------------+

# ------------------- Visualization -------------------
# feel free to comment out and construct your own plots.
m_light_red  = label >= 1
m_dark_red   = (label >= 0) & (label < 1)
m_dark_blue  = (label < 0)  & (label > -1)
m_light_blue = label <= -1
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
#plt.savefig('p2q3.png', dpi=1200)
plt.show()