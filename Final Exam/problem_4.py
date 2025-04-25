import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from IPython.core.display import HTML
from sympy import Matrix
import matplotlib.tri as tri


"""
Given Values
"""
E = 3E7
t_bar = 0
t_bar_y = -20
v = 0.3

"""
Calculate the D matrix
"""

D = E/(1-v**2)*sp.Matrix([[1, v, 0], [v, 1, 0], [0, 0, (1-v)/2]])

print("D Matrix")
sp.pprint(D)


"""
Set up the Coordinate Matrix
In the example we only needed one coordinate system since we were dealing with a single quadrillateral
Now we need two as we are dealing with two triangles
"""

# Split the quadrilateral into two triangles
triangle_1 = sp.Matrix([
    [0, 1],  # Node 2
    [0, 0],  # Node 1
    [2, 0.5] # Node 3
])

triangle_2 = sp.Matrix([
    [0, 1],  # Node 2
    [2, 0.5],# Node 3
    [2, 1]   # Node 4
])

# Print the coordinates for verification
print("Triangle 1 Coordinates (Nodes 2, 1, 3):")
sp.pprint(triangle_1)

print("\nTriangle 2 Coordinates (Nodes 2, 3, 4):")
sp.pprint(triangle_2)


# Print the coordinates for verification
print("Triangle 1 Coordinates:")
sp.pprint(triangle_1)

print("\nTriangle 2 Coordinates:")
sp.pprint(triangle_2)


'''
Create the Shape Functions
'''
# triangular shape functions

xi, eta = sp.symbols('xi eta')

N1 = 1 - xi - eta 
N2 = xi
N3 = eta

'''
Compute the Jacobian Matrix
'''

# Shape function derivatives
dN1dxi = sp.diff(N1, xi)
dN2dxi = sp.diff(N2, xi)
dN3dxi = sp.diff(N3, xi)
dN1deta = sp.diff(N1, eta)
dN2deta = sp.diff(N2, eta)
dN3deta = sp.diff(N3, eta)

# Derivatives of shape functions with respect to xi and eta
dN_dxi = sp.Matrix([dN1dxi, dN2dxi, dN3dxi])   # [∂N1/∂xi, ∂N2/∂xi, ∂N3/∂xi]
dN_deta = sp.Matrix([dN1deta, dN2deta, dN3deta]) # [∂N1/∂eta, ∂N2/∂eta, ∂N3/∂eta]

# Combine derivatives into a single matrix
dN = sp.Matrix.hstack(dN_dxi, dN_deta)  # Shape function derivative matrix

# Compute the Jacobian matrix
J_triangle1 = dN.T * triangle_1

print("Jacobian Matrix for Triangle 1:")
sp.pprint(J_triangle1)


J_triangle2 = dN.T * triangle_2

print("\nJacobian Matrix for Triangle 2:")
sp.pprint(J_triangle2)

'''
Determinant of the Jacobian Matrix
Confirm that the determinant is not equal to zero

Then I took the inverse of the Jacobian Matrix
'''
det_J_triangle1 = sp.det(J_triangle1)

print("Determinant of the Jacobian Matrix of triangle 1:")
sp.pprint(det_J_triangle1)

inv_J_triangle1 = J_triangle1.inv()

print("Inverse of the Jacobian Matrix of triangle 1:")
sp.pprint(inv_J_triangle1)

det_J_triangle2 = sp.det(J_triangle2)

print("Determinant of the Jacobian Matrix of triangle 2:")
sp.pprint(det_J_triangle2)

inv_J_triangle2 = J_triangle2.inv()

print("Inverse of the Jacobian Matrix of triangle 2:")
sp.pprint(inv_J_triangle2)


'''
Calculate the B matrix for both triangles
'''

# Derivatives of shape functions in parent coordinates
dN_parent = sp.Matrix([
    [-1, 1, 0],  # ∂N/∂xi
    [-1, 0, 1]   # ∂N/∂eta
])

# Compute derivatives in global coordinates
global_derivatives = inv_J_triangle1 * dN_parent

# Construct the B-matrix for Triangle 1
B1 = sp.zeros(3, 6)  # Initialize a 3x6 matrix

# Assign ∂N/∂x
B1[0, 0] = global_derivatives[0, 0]
B1[0, 2] = global_derivatives[0, 1]
B1[0, 4] = global_derivatives[0, 2]

# Assign ∂N/∂y
B1[1, 1] = global_derivatives[1, 0]
B1[1, 3] = global_derivatives[1, 1]
B1[1, 5] = global_derivatives[1, 2]

# Assign ∂N/∂y and ∂N/∂x for the shear terms
B1[2, 0] = global_derivatives[1, 0]
B1[2, 2] = global_derivatives[1, 1]
B1[2, 4] = global_derivatives[1, 2]
B1[2, 1] = global_derivatives[0, 0]
B1[2, 3] = global_derivatives[0, 1]
B1[2, 5] = global_derivatives[0, 2]


print("Strain-Displacement Matrix (B) for Triangle 1:")
sp.pprint(B1)


# Compute derivatives in global coordinates for Triangle 2
global_derivatives2 = inv_J_triangle2 * dN_parent

# Construct the B-matrix for Triangle 2
B2 = sp.zeros(3, 6)  # Initialize a 3x6 matrix

# Assign ∂N/∂x
B2[0, 0] = global_derivatives2[0, 0]
B2[0, 2] = global_derivatives2[0, 1]
B2[0, 4] = global_derivatives2[0, 2]

# Assign ∂N/∂y
B2[1, 1] = global_derivatives2[1, 0]
B2[1, 3] = global_derivatives2[1, 1]
B2[1, 5] = global_derivatives2[1, 2]

# Assign ∂N/∂y and ∂N/∂x for the shear terms
B2[2, 0] = global_derivatives2[1, 0]
B2[2, 2] = global_derivatives2[1, 1]
B2[2, 4] = global_derivatives2[1, 2]
B2[2, 1] = global_derivatives2[0, 0]
B2[2, 3] = global_derivatives2[0, 1]
B2[2, 5] = global_derivatives2[0, 2]

print("\nStrain-Displacement Matrix (B) for Triangle 2:")
sp.pprint(B2)

'''
Calculate the global stiffness matrix
'''

# Define the Gauss quadrature integration weight and point
w = 1  # Weight for a single integration point
xi, eta = 1/3, 1/3  # Barycentric coordinates for the integration point

# Element stiffness matrix for Triangle 1
K1 = (B1.T * D * B1) * det_J_triangle1 * w
print("\nElement Stiffness Matrix for Triangle 1:")
sp.pprint(K1)

# Element stiffness matrix for Triangle 2
K2 = (B2.T * D * B2) * det_J_triangle2 * w
print("\nElement Stiffness Matrix for Triangle 2:")
sp.pprint(K2)


# Initialize the global stiffness matrix (6 DOF for 4 nodes)
K_global = sp.zeros(8, 8)

# Assembly for Triangle 1 (Nodes 2, 1, 3 -> Global DOFs [2, 3, 0, 1, 4, 5])
node_map1 = [2, 3, 0, 1, 4, 5]
for i in range(6):
    for j in range(6):
        K_global[node_map1[i], node_map1[j]] += K1[i, j]

# Assembly for Triangle 2 (Nodes 2, 3, 4 -> Global DOFs [2, 3, 4, 5, 6, 7])
node_map2 = [2, 3, 4, 5, 6, 7]
for i in range(6):
    for j in range(6):
        K_global[node_map2[i], node_map2[j]] += K2[i, j]


print("\nGlobal Stiffness Matrix:")
sp.pprint(K_global)

'''
Apply the boundary conditions
'''
# Initialize the global force vector (8 DOFs for 4 nodes)
F_global = sp.zeros(8, 1)

# Apply the boundary conditions
constrained_dofs = [0, 1]  # DOFs for Node 1 (fixed at x=0, y=0)

# Modify the global stiffness matrix and force vector for the constrained DOFs
for dof in constrained_dofs:
    K_global[dof, :] = sp.zeros(1, K_global.shape[1])  # Zero out the entire row
    K_global[:, dof] = sp.zeros(K_global.shape[0], 1)  # Zero out the entire column
    K_global[dof, dof] = 1  # Set diagonal to 1 to maintain structure
    F_global[dof] = 0  # Set force at constrained DOF to zero

# Print the modified global stiffness matrix and force vector
print("\nModified Global Stiffness Matrix:")
sp.pprint(K_global)

print("\nModified Global Force Vector:")
sp.pprint(F_global)

# Define edge length and traction force
t_y = -20  # Traction in the y-direction
L_edge = sp.sqrt((2 - 2)**2 + (1 - 0.5)**2)  # Length of edge between Node 3 and Node 4 (0.5)

# Shape functions along the edge
eta = sp.symbols('eta')  # Natural coordinate along the edge
N3_edge = (1 - eta) / 2  # Shape function for Node 3
N4_edge = (1 + eta) / 2  # Shape function for Node 4

# Integrate the force contributions
F3 = sp.integrate(N3_edge * t_y * L_edge / 2, (eta, -1, 1))
F4 = sp.integrate(N4_edge * t_y * L_edge / 2, (eta, -1, 1))

# Initialize the global force vector (8 DOFs for 4 nodes)
F_global = sp.zeros(8, 1)

# Assign contributions to the global force vector
F_global[4] += F3  # y-direction DOF of Node 3
F_global[6] += F4  # y-direction DOF of Node 4

# Print the updated global force vector
print("\nUpdated Global Force Vector with Traction Contribution:")
sp.pprint(F_global)

'''
Since xi is fixed for the traction force and nodes 3 and 4 are on the same edge, I can simplify the matrices
So I set up a reduced matrix for the k AND d values
'''

# Extract the bottom-right 4x4 section
K_reduced = K_global[-4:, -4:]

# Print the extracted matrix
print("\nReduced Global Stiffness Matrix (K_reduced):")
sp.pprint(K_reduced)

dy = sp.Matrix([sp.symbols('ux3'), sp.symbols('uy3'), sp.symbols('ux4'), sp.symbols('uy4')])

f_reduced = sp.Matrix([-5,0,-5,0])

'''
Solve the reduced system
'''

d_reduced = K_reduced.LUsolve(f_reduced)

print("\nDisplacements for Unconstrained Nodes:")
sp.pprint(d_reduced)

'''
Calculate the resulting strains and stresses
'''

# Extract displacements for Triangle 1 (Nodes 1, 2, 3 -> DOFs [0, 1, 2, 3, 4, 5])
d1 = sp.Matrix([
    0, 0,  # Node 1 (fixed)
    d_reduced[0], d_reduced[1],  # Node 2
    d_reduced[2], d_reduced[3]   # Node 3
])

# Compute strain for Triangle 1
strain_1 = B1 * d1
print("\nStrain for Triangle 1:")
sp.pprint(strain_1)

# Compute stress for Triangle 1
stress_1 = D * strain_1
print("\nStress for Triangle 1:")
sp.pprint(stress_1)

# Extract displacements for Triangle 2 (Nodes 2, 3, 4 -> DOFs [2, 3, 4, 5, 6, 7])
d2 = sp.Matrix([
    d_reduced[0], d_reduced[1],  # Node 2
    d_reduced[2], d_reduced[3],  # Node 3
    0, 0  # Node 4 (fixed)
])

# Compute strain for Triangle 2
strain_2 = B2 * d2
print("\nStrain for Triangle 2:")
sp.pprint(strain_2)

# Compute stress for Triangle 2
stress_2 = D * strain_2
print("\nStress for Triangle 2:")
sp.pprint(stress_2)


'''
Code that attempts to get the stress and strain values for each node separately
'''

# Gauss Point for 1-point integration
xi_eta_points = [(1/3, 1/3)]  # You can add more points for higher-order integration
weights = [1]  # Corresponding weights

# Loop over Gauss points for Triangle 1
print("\nStrains and Stresses for Triangle 1:")
for (xi, eta), w in zip(xi_eta_points, weights):
    # Evaluate the Jacobian and its determinant at this Gauss point
    J_eval = J_triangle1.subs({'xi': xi, 'eta': eta})
    det_J_eval = det_J_triangle1.subs({'xi': xi, 'eta': eta})
    inv_J_eval = J_eval.inv()

    # Compute derivatives in global coordinates
    global_derivatives = inv_J_eval * dN_parent

    # Construct the B-matrix for this Gauss point
    B_gauss = sp.zeros(3, 6)
    B_gauss[0, 0] = global_derivatives[0, 0]
    B_gauss[0, 2] = global_derivatives[0, 1]
    B_gauss[0, 4] = global_derivatives[0, 2]
    B_gauss[1, 1] = global_derivatives[1, 0]
    B_gauss[1, 3] = global_derivatives[1, 1]
    B_gauss[1, 5] = global_derivatives[1, 2]
    B_gauss[2, 0] = global_derivatives[1, 0]
    B_gauss[2, 2] = global_derivatives[1, 1]
    B_gauss[2, 4] = global_derivatives[1, 2]
    B_gauss[2, 1] = global_derivatives[0, 0]
    B_gauss[2, 3] = global_derivatives[0, 1]
    B_gauss[2, 5] = global_derivatives[0, 2]

    # Compute strain at this Gauss point
    strain_gauss = B_gauss * d1
    print(f"\nStrain at Gauss Point ({xi}, {eta}):")
    sp.pprint(strain_gauss)

    # Compute stress at this Gauss point
    stress_gauss = D * strain_gauss
    print(f"\nStress at Gauss Point ({xi}, {eta}):")
    sp.pprint(stress_gauss)

# Repeat for Triangle 2
print("\nStrains and Stresses for Triangle 2:")
for (xi, eta), w in zip(xi_eta_points, weights):
    # Evaluate the Jacobian and its determinant at this Gauss point
    J_eval = J_triangle2.subs({'xi': xi, 'eta': eta})
    det_J_eval = det_J_triangle2.subs({'xi': xi, 'eta': eta})
    inv_J_eval = J_eval.inv()

    # Compute derivatives in global coordinates
    global_derivatives = inv_J_eval * dN_parent

    # Construct the B-matrix for this Gauss point
    B_gauss = sp.zeros(3, 6)
    B_gauss[0, 0] = global_derivatives[0, 0]
    B_gauss[0, 2] = global_derivatives[0, 1]
    B_gauss[0, 4] = global_derivatives[0, 2]
    B_gauss[1, 1] = global_derivatives[1, 0]
    B_gauss[1, 3] = global_derivatives[1, 1]
    B_gauss[1, 5] = global_derivatives[1, 2]
    B_gauss[2, 0] = global_derivatives[1, 0]
    B_gauss[2, 2] = global_derivatives[1, 1]
    B_gauss[2, 4] = global_derivatives[1, 2]
    B_gauss[2, 1] = global_derivatives[0, 0]
    B_gauss[2, 3] = global_derivatives[0, 1]
    B_gauss[2, 5] = global_derivatives[0, 2]

    # Compute strain at this Gauss point
    strain_gauss = B_gauss * d2
    print(f"\nStrain at Gauss Point ({xi}, {eta}):")
    sp.pprint(strain_gauss)

    # Compute stress at this Gauss point
    stress_gauss = D * strain_gauss
    print(f"\nStress at Gauss Point ({xi}, {eta}):")
    sp.pprint(stress_gauss)
