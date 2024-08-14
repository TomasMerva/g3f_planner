import numpy as np
import scipy as sp
from scipy import sparse
import osqp
import casadi as ca

# N = 4
# num_dofs = 2
# xr = np.array([-1,-2])
# x0 = np.array([-5,-5])

# Qq = np.eye(num_dofs) # only defined for one waypoint
# Q = np.zeros((num_dofs, num_dofs)) # only defined for one waypoint

# # quadratic objective
# P = sparse.block_diag([sparse.kron(sparse.eye(N), Q)], format='csc') # shape: [N, ndofs]

# # linear objective
# q = np.hstack([np.kron(np.ones(N), -2*(Qq@xr))]) # maybe replaced by initial guess?? its just need to be vector

# prob = osqp.OSQP()
# # Add constraints (optional)
# A = sparse.eye(N * num_dofs)  # Example: identity matrix for constraints
# l = np.ones(N * num_dofs) * -10  # Lower bounds
# u = np.ones(N * num_dofs) * 10   # Upper bounds

# prob.setup(P, q, A, l, u)

# x_init = np.tile(x0, N)
# print("x_init:", x_init)
# prob.warm_start(x_init)
# res = prob.solve()

# print(res.x)


# obj_eval = 

# import numpy as np
# import scipy as sp
# from scipy import sparse
# import osqp

def create_P_matrix(num_waypoints, num_dof):
    FD_matrix = np.zeros((num_waypoints, num_waypoints), dtype=float)
    FD_INDICES = [1., -2., 1.]
    for i in range(1, num_waypoints-1):
        FD_matrix[i, (i-1):(i-1)+3] = FD_INDICES
    FD_matrix[0,0]=1
    FD_matrix[-1,-1] = 1
    R = FD_matrix.T@FD_matrix
    return sparse.csc_matrix(sparse.kron(R, sparse.eye(num_dof)))

# Number of variables (N time steps, 2 DOFs)
N = 4  # Number of time steps
num_dofs = 2  # Degrees of freedom

# Reference point (where we want x to be close to)
xr = np.array([1, 2])

# Constructing the quadratic cost matrix P
Q = np.eye(num_dofs)  # Identity matrix for each degree of freedom

QN =create_P_matrix(N, num_dofs)
P = sparse.kron(sparse.eye(N), Q) + QN

print(P.toarray())
# Constructing the linear cost vector q
# q should reflect the linear cost, which helps in achieving the reference state
q = -np.hstack([np.kron(np.ones(N), xr)])
print(q)
# Constraints: set reasonable bounds
A = sparse.eye(N * num_dofs)
l = np.ones(N * num_dofs) * -10  # Lower bounds
u = np.ones(N * num_dofs) * 10   # Upper bounds

# Setup OSQP problem
prob = osqp.OSQP()
prob.setup(P, q, A, l, u)

# Solve the problem
res = prob.solve()

# Display the results
print("Optimal Solution:", res.x)
print("Difference from Reference Point:", res.x - np.tile(xr, N))
