import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from IPython.core.display import HTML
import matplotlib.tri as tri
from sympy import Matrix

'''
Construct the Conduction Matrix
'''

'''
Step 1 Create the Element Matrix
'''
# Corrected element matrix to reflect the actual node positions and counterclockwise ordering
E = Matrix([[0, 0], [0.5, 0], [0.5, 0.5]])  # Node A (0,0), Node C (0.5, 0), Node B (0.5, 0.5)

'''
 Step 2: Define the Shape Functions
'''

# Natural coordinates for the triangular element
xi, eta = sp.symbols('xi eta')

# Define shape functions for triangular element
N1 = 1 - xi - eta
N2 = xi
N3 = eta
N_tri = Matrix([N1, N2, N3])

'''
Step 3: The Gradient
'''
# Calculate partial derivatives of triangular shape functions
partial_xi = [sp.diff(Ni, xi) for Ni in N_tri]
partial_eta = [sp.diff(Ni, eta) for Ni in N_tri]

# Define the gradient matrix G for the triangular element
G = Matrix([partial_xi, partial_eta])

'''
Step 4: The Jacobian
'''

# Calculate the Jacobian matrix by multiplying G with the coordinates of each node (E)
J = G * E

'''
Step 5: Determinant and Inverse of the Jacobian
'''

# Calculate the determinant and inverse of the Jacobian
J_det = J.det()
J_inv = J.inv()

print("\nJacobian Matrix J:")
print(J)
print("\nDeterminant of Jacobian |J|:")
print(J_det)
print("\nInverse of Jacobian J^-1:")
print(J_inv)

'''
Step 6: Derivatives of the Shape Functions (B)
'''

# Define material property: thermal conductivity
k = 2  # Thermal conductivity (W/m/degC)
D = k  # Isotropic material, so D is a scalar

B = J_inv * G

# Compute B^T * D * B and integrate over natural coordinates
B_T_D_B = B.T * D * B
K_element = sp.integrate(sp.integrate(B_T_D_B * J_det, (xi, 0, 1 - eta)), (eta, 0, 1))

print("\nB Matrix (Derivatives of Shape Functions in Global Coordinates):")
print(B)
print("\nConductance Matrix K for the Triangular Element:")
print(K_element)

'''
Construct the boundary Flux Matrix resulting from the flux acting on the edges x = 0.5 and y = x
'''

# Flux (q) is applied along x = 0.5 and y = x, which are the boundaries of the triangular element
q = 10  # Flux value (W/m)

# To calculate the flux matrix, substitute appropriate boundary values into shape functions
# Correcting the integration limits for proper representation in natural coordinates
N_boundary_x = N_tri.subs({xi: 0.5})
N_boundary_y = N_tri.subs({eta: xi})

# Integrate along the boundaries to calculate the flux matrix
f_gamma_x = sp.integrate(q * N_boundary_x.T, (eta, 0, 0.5))
f_gamma_y = sp.integrate(q * N_boundary_y.T, (xi, 0, 0.5))

# Add contributions from both boundaries
f_gamma = f_gamma_x + f_gamma_y

print("\nBoundary Flux Matrix f_gamma:")
print(f_gamma)

''' 
Construct the Source Matrix consisting of uniformly distributed source s = 10
and the point source P = 7.
'''

# Given a constant source of s = 10 W/m^2 and point source P = 7 W at origin
s = 10
P = 7

# Calculate the source matrix due to distributed source
f_Omega = sp.integrate(sp.integrate(s * N_tri.T * J_det, (xi, 0, 1 - eta)), (eta, 0, 1))

# Add point source contribution
# Node A is located at the origin, so P = 7 W acts directly at Node A (node 1)
f_Omega = f_Omega.tolist()  # Convert to list for modification
f_Omega[0][0] += P  # Correct indexing to properly add point source
f_Omega = Matrix(f_Omega)  # Convert back to Matrix

print("\nUpdated Source Matrix f_Omega:")
print(f_Omega)

'''
Calculate the Unknown Temperature Matrix
'''

# Convert matrices to numerical form for calculations
K_element_numeric = np.array(K_element).astype(np.float64)
f_gamma_numeric = np.array(f_gamma).astype(np.float64).reshape(-1, 1)  # Ensure column vector shape
f_Omega_numeric = np.array(f_Omega).astype(np.float64).reshape(-1, 1)  # Ensure column vector shape

# Combine flux and source contributions
F_numeric = f_gamma_numeric + f_Omega_numeric  # This will be a (3, 1) vector

# Known temperatures at nodes A and B
T_known = 5  # Temperature at nodes A and B is 5°C

# Extract parts of K and F related to the unknown temperature at Node C
# Since both Nodes A and B have known temperatures, we only need to solve for Node C
K_unknown = K_element_numeric[2, 2]  # Extract the single entry for Node C
F_unknown = F_numeric[2] - (K_element_numeric[2, 0] * T_known + K_element_numeric[2, 1] * T_known)

# Solve for the unknown temperature at Node C
T_unknown = F_unknown / K_unknown

# Convert T_unknown from an array to a scalar value using .item()
T_C = T_unknown.item()

# Display the result
print("\nTemperature at Node C: {:.2f} °C".format(T_C))

'''
Calculate the Reaction Forces at Nodes A and B
'''

# Form the full temperature vector
T_full = np.array([T_known, T_known, T_C]).reshape(-1, 1)

# Calculate the reaction forces using the global stiffness matrix
F_reaction = K_element_numeric @ T_full

# Display the reaction forces at Nodes A and B
print("\nReaction at Node A: {:.2f} W".format(F_reaction[0, 0]))
print("Reaction at Node B: {:.2f} W".format(F_reaction[1, 0]))

