import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from IPython.core.display import HTML

'''
Set up our given variables
'''

k = 4
T = 10
q_0 = 0
q_1 = 30  # Flux value at the boundary
s = 50

'''
Step 1: The Element Matrix
'''
# I set up the element matrix based on the nodal positions. Node 1 is at point b and the nodes
# continue counter clockwise. Therefore node 1 is at (2,0), node 2 is at (2,1), node 3 is at (0,1), and node 4 is at (0,0)

E = sp.Matrix([[2, 0], [2, 1], [0, 1], [0, 0]])

'''
Step 2: The Gradient
'''
# I set up x and n for xi and eta respectively
x, n = sp.symbols('x n')

# The gradient is the partial derivative of the shape functions with respect to x and n
GN = (1/4) * sp.Matrix([[n-1, 1-n, 1+n, -n-1], [x-1, -x-1, 1+x, 1-x]])

'''
Step 3: The Jacobian
'''
# Multiply G and E to get the Jacobian
J = GN * E

'''
Step 4: Determinant and inverse of the Jacobian
'''
# Calculate the determinant of the Jacobian
det_J = J.det()

# calculate the inverse of the Jacobian
J_inv = J.inv()


'''
Step 5: Derivatives of the Shape Functions (B)
'''
# To get the derivative of the Shape Function (B) multiply the inverse with Gradient
B = J_inv * GN

'''
Step 6: The Conduction Matrix
'''

K = sp.integrate(B.T * B * det_J, (x, -1, 1), (n, -1, 1))
print("\nConduction Matrix K:")
print(K)

'''
Step 7: The Source Matrix
'''

# I have not created an element matrix yet, so I am defining the shape functions here
N_1 = (1/4) * (1-x) * (1-n)
N_2 = (1/4) * (1+x) * (1-n)
N_3 = (1/4) * (1+x) * (1+n)
N_4 = (1/4) * (1-x) * (1+n)

# Now I put the shape functions into a matrix
N = sp.Matrix([N_1, N_2, N_3, N_4])

# Now I can calculate the source matrix
f_Omega = sp.integrate(s * N.T * det_J, (x, -1, 1), (n, -1, 1))

print("\nSource Matrix f_Omega:")
print(f_Omega)

'''
Step 8: The Boundary Flux Matrix (f_gamma)
'''
# The problem states that the flux is 0 along AB and AD
# However the flux is 30 along DC. DC is in the n direction along the edge n=1
# I set n = 1 which is now a constant and can integrate along x next.
N_boundary = N.subs(n, 1)

# Integrate along x from -1 to 1
f_gamma = - sp.integrate(q_1 * N_boundary.T, (x, -1, 1))  # q_1 = 30


print("\nBoundary Flux Matrix f_gamma:")
print(f_gamma)

'''
Step 9: RHS Matrix
'''
# The right hand side can be calculated with f_Omega + f_gamma + R
# R is the reaction term which is 0 in this case so I neglected it
RHS = f_Omega + f_gamma

# Convert RHS to a column vector by transposing
RHS = RHS.T

# Print the RHS Matrix for verification
print("\nRHS Matrix (Column Vector):")
print(RHS)

'''
Step 10: Assemble the Global Stiffness Matrix and Apply Boundary Conditions
'''

# The element stiffness matrix represents how heat travels through the element
# The shape function and determinant of the Jacobian are used to calculate the element stiffness matrix
# This gives us the way the heat travels through the element due to the shape of the element
# But we also need to consider the material of the element which is represented by the conductivity matrix
# The conductivity matrix is multiplied by the element stiffness matrix to get the element matrix

K_global = k * K

# Convert the stiffness matrix and source terms to numeric
K_global_numeric = sp.Matrix(K_global)
f_Omega_numeric = sp.Matrix(f_Omega)
f_gamma_numeric = sp.Matrix(f_gamma)

# I know the T boundary condition at nodes 1 and 2 (T1 = T2 = 10°C)
# I want to set those values for Nodes 1 and 2
# Convert the matrices to lists for modification

K_modified = K_global_numeric.tolist()
RHS_modified = RHS.tolist()

# Nodes 1 and 2 correspond to indices 0 and 1 in the Python list (0-based index)
node_1_index = 0
node_2_index = 1

# I need to modify the stiffness matrix and RHS vector to account for the boundary conditions

# Set row 0 of K to zero and set diagonal to 1 (Node 1)
K_modified[node_1_index] = [0] * len(K_modified)
K_modified[node_1_index][node_1_index] = 1
# Set corresponding RHS value to prescribed temperature (10°C)
RHS_modified[node_1_index] = [10]

# Set row 1 of K to zero and set diagonal to 1 (Node 2)
K_modified[node_2_index] = [0] * len(K_modified)
K_modified[node_2_index][node_2_index] = 1
# Set corresponding RHS value to prescribed temperature (10°C)
RHS_modified[node_2_index] = [10]

# Convert back to sympy Matrix for solving
K_modified_matrix = sp.Matrix(K_modified)
RHS_modified_matrix = sp.Matrix(RHS_modified)

print("\nModified Stiffness Matrix K:")
print(K_modified_matrix)

print("\nModified RHS Vector:")
print(RHS_modified_matrix)

'''
Step 14: Solve for Nodal Temperatures
'''

# Solve the matrix equation to get the nodal temperatures
# I know that node 1 and 2 are 10°C so I can solve for the other nodes
nodal_temperatures = K_modified_matrix.LUsolve(RHS_modified_matrix)
print("\nNodal Temperatures:")
print(nodal_temperatures)

'''
Step 15: Calculate Nodal Fluxes
'''
# The nodal fluxes can be determined by multiplying the original K matrix by the nodal temperatures
nodal_fluxes = K_global_numeric * nodal_temperatures - RHS
print("\nNodal Fluxes:")
print(nodal_fluxes)


