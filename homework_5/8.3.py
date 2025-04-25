import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from IPython.core.display import HTML
import matplotlib.tri as tri

'''
Step 1: The Element Matrix
'''

# Define coordinates of each node in the element
E1 = sp.Matrix([[0, 0],  # Node 1
                [2, 0],  # Node 2
                [2, 2],  # Node 3
                [0, 2]]) # Node 5

'''
Step 2: Find the Shape Functions
'''
# I decided to set up the shape funcitons first this time. I skipped that step in 8.1 and had to go back

# Define natural coordinates (xi, eta)
xi, eta = sp.symbols('xi eta')

# Define shape functions for a four-node rectangular element
N1 = (1/4) * (1 - xi) * (1 - eta)
N2 = (1/4) * (1 + xi) * (1 - eta)
N3 = (1/4) * (1 + xi) * (1 + eta)
N4 = (1/4) * (1 - xi) * (1 + eta)

# Collect shape functions into a matrix
N = sp.Matrix([N1, N2, N3, N4])

'''
Step 3: The Gradient
'''

# Calculate partial derivatives of shape functions with respect to xi and eta
dN_dxi = [sp.diff(Ni, xi) for Ni in N]
dN_deta = [sp.diff(Ni, eta) for Ni in N]

# Define the gradient matrix G
G = sp.Matrix([dN_dxi, dN_deta])

print("\nGradient Matrix G:")
print(G)

'''
Step 4: The Jacobian
'''

# Multiply G and E1 to get the Jacobian matrix
J = G * E1

print("\nJacobian Matrix J:")
print(J)

'''
Step 5: Determinant and Inverse of the Jacobian
'''

# Calculate the determinant of the Jacobian
det_J = J.det()

# Calculate the inverse of the Jacobian
J_inv = J.inv()

print("\nDeterminant of the Jacobian (det_J):")
print(det_J)

print("\nInverse of the Jacobian (J_inv):")
print(J_inv)

'''
Step 6: Derivatives of the Shape Functions (B)
'''

# Calculate the B matrix (derivative of the shape functions with respect to global coordinates)
B = J_inv * G

print("\nB Matrix (Derivatives of Shape Functions in Global Coordinates):")
print(B)

'''
Step 7: The Conduction Matrix
'''

# Define the thermal conductivity for Element 1
k_1 = 1

# The material property matrix D for isotropic material (scalar k)
D = k_1

# Calculate the B^T * D * B product
BT_D_B = B.T * D * B

# Now we integrate over natural coordinates xi and eta, from -1 to 1
K_element1 = sp.integrate(sp.integrate(BT_D_B * det_J, (xi, -1, 1)), (eta, -1, 1))

print("\nConduction Matrix K for Element 1:")
print(K_element1)


# Substitute numerical values for k_2 and det_J
k_2 = 2
det_J = 4
D = k_2

# Calculate the B^T * D * B product
BT_D_B = B.T * D * B

# Calculate the conduction matrix K for element 2
K_element2 = sp.integrate(sp.integrate(BT_D_B * det_J, (xi, -1, 1)), (eta, -1, 1))


print("\nConduction Matrix K for Element 2:")
print(K_element2)


import numpy as np

# Number of nodes in the global system
num_nodes = 5

# Initialize the global stiffness matrix
K_global = np.zeros((num_nodes, num_nodes))

# Element stiffness matrices (manually copied from earlier results)
K_element1 = np.array([[0.6667, -0.1667, -0.3333, -0.1667],
                       [-0.1667, 0.6667, -0.1667, -0.3333],
                       [-0.3333, -0.1667, 0.6667, -0.1667],
                       [-0.1667, -0.3333, -0.1667, 0.6667]])

K_element2 = np.array([[5.333, -1.333, -2.667, -1.333],
                       [-1.333, 5.333, -1.333, -2.667],
                       [-2.667, -1.333, 5.333, -1.333],
                       [-1.333, -2.667, -1.333, 5.333]])

# Node mappings for each element
element1_nodes = [0, 1, 2, 4]  # Nodes 1, 2, 3, 5 (Python indexing: 0-based)
element2_nodes = [2, 3, 4]      # Nodes 3, 4, 5 (Python indexing: 0-based)

# Assemble Element 1 into the global stiffness matrix
for i in range(len(element1_nodes)):
    for j in range(len(element1_nodes)):
        K_global[element1_nodes[i], element1_nodes[j]] += K_element1[i, j]

# Assemble Element 2 into the global stiffness matrix
for i in range(len(element2_nodes)):
    for j in range(len(element2_nodes)):
        K_global[element2_nodes[i], element2_nodes[j]] += K_element2[i, j]

# Print the global stiffness matrix
print("\nGlobal Stiffness Matrix K_global:")
print(K_global)


# -----------------------------
# Applying Boundary Conditions
# -----------------------------

# Define the load vector (initially set to zero)
F = np.zeros(num_nodes)

# Apply point source at node 5 (F5 = 10 W)
F[4] = 10

# Prescribed temperatures at nodes 1 and 2 (T1 = T2 = 10 C)
prescribed_nodes = [0, 1]
prescribed_temperature = 10

# Apply the boundary conditions for prescribed temperatures
for node in prescribed_nodes:
    # Set corresponding row and column in K_global to zero
    K_global[node, :] = 0
    K_global[:, node] = 0
    
    # Set the diagonal element to 1
    K_global[node, node] = 1
    
    # Set the corresponding entry in the load vector to the prescribed temperature
    F[node] = prescribed_temperature

print("\nGlobal Stiffness Matrix K_global after applying boundary conditions:")
print(K_global)
print("\nLoad Vector F after applying boundary conditions:")
print(F)

import sympy as sp
from sympy import Matrix

'''
Step 1: The Element Matrix
'''

# Define coordinates of each node in the element
E1 = Matrix([[0, 0],  # Node 1
             [2, 0],  # Node 2
             [2, 2],  # Node 3
             [0, 2]]) # Node 5

'''
Step 2: Find the Shape Functions
'''
# Define natural coordinates (xi, eta)
xi, eta = sp.symbols('xi eta')

# Define shape functions for a four-node rectangular element
N1 = (1/4) * (1 - xi) * (1 - eta)
N2 = (1/4) * (1 + xi) * (1 - eta)
N3 = (1/4) * (1 + xi) * (1 + eta)
N4 = (1/4) * (1 - xi) * (1 + eta)

# Collect shape functions into a matrix
N = Matrix([N1, N2, N3, N4])

'''
Step 3: The Gradient
'''

# Calculate partial derivatives of shape functions with respect to xi and eta
dN_dxi = [sp.diff(Ni, xi) for Ni in N]
dN_deta = [sp.diff(Ni, eta) for Ni in N]

# Define the gradient matrix G
G = Matrix([dN_dxi, dN_deta])

print("\nGradient Matrix G:")
print(G)

'''
Step 4: The Jacobian
'''

# Multiply G and E1 to get the Jacobian matrix
J = G * E1

print("\nJacobian Matrix J:")
print(J)

'''
Step 5: Determinant and Inverse of the Jacobian
'''

# Calculate the determinant of the Jacobian
det_J = J.det()

# Calculate the inverse of the Jacobian
J_inv = J.inv()

print("\nDeterminant of the Jacobian (det_J):")
print(det_J)

print("\nInverse of the Jacobian (J_inv):")
print(J_inv)

'''
Step 6: Derivatives of the Shape Functions (B)
'''

# Calculate the B matrix (derivative of the shape functions with respect to global coordinates)
B = J_inv * G

print("\nB Matrix (Derivatives of Shape Functions in Global Coordinates):")
print(B)

'''
Step 7: The Conduction Matrix
'''

# Define the thermal conductivity for Element 1
k_1 = 1

# The material property matrix D for isotropic material (scalar k)
D = k_1

# Calculate the B^T * D * B product
BT_D_B = B.T * D * B

# Now we integrate over natural coordinates xi and eta, from -1 to 1
K_element1 = sp.integrate(sp.integrate(BT_D_B * det_J, (xi, -1, 1)), (eta, -1, 1))

print("\nConduction Matrix K for Element 1:")
print(K_element1)

# Substitute numerical values for k_2 and det_J
k_2 = 2
det_J = 4
D = k_2

# Calculate the B^T * D * B product
BT_D_B = B.T * D * B

# Calculate the conduction matrix K for element 2
K_element2 = sp.integrate(sp.integrate(BT_D_B * det_J, (xi, -1, 1)), (eta, -1, 1))

print("\nConduction Matrix K for Element 2:")
print(K_element2)

'''
Step 8: Global Stiffness Matrix Assembly
'''

# Number of nodes in the global system
num_nodes = 5

# Initialize the global stiffness matrix
K_global = Matrix.zeros(num_nodes, num_nodes)

# Element stiffness matrices (converted from earlier symbolic matrices)
K_element1 = Matrix([[0.6667, -0.1667, -0.3333, -0.1667],
                     [-0.1667, 0.6667, -0.1667, -0.3333],
                     [-0.3333, -0.1667, 0.6667, -0.1667],
                     [-0.1667, -0.3333, -0.1667, 0.6667]])

K_element2 = Matrix([[5.333, -1.333, -2.667, -1.333],
                     [-1.333, 5.333, -1.333, -2.667],
                     [-2.667, -1.333, 5.333, -1.333],
                     [-1.333, -2.667, -1.333, 5.333]])

# Node mappings for each element
element1_nodes = [0, 1, 2, 4]  # Nodes 1, 2, 3, 5 (Python indexing: 0-based)
element2_nodes = [2, 3, 4]      # Nodes 3, 4, 5 (Python indexing: 0-based)

# Assemble Element 1 into the global stiffness matrix
for i in range(len(element1_nodes)):
    for j in range(len(element1_nodes)):
        K_global[element1_nodes[i], element1_nodes[j]] += K_element1[i, j]

# Assemble Element 2 into the global stiffness matrix
for i in range(len(element2_nodes)):
    for j in range(len(element2_nodes)):
        K_global[element2_nodes[i], element2_nodes[j]] += K_element2[i, j]

# Print the global stiffness matrix
print("\nGlobal Stiffness Matrix K_global:")
print(K_global)

# -----------------------------
# Applying Boundary Conditions
# -----------------------------

# Define the load vector (initially set to zero)
F = Matrix.zeros(num_nodes, 1)

# Apply point source at node 5 (F5 = 10 W)
F[4] = 10

# Prescribed temperatures at nodes 1 and 2 (T1 = T2 = 10 C)
prescribed_nodes = [0, 1]
prescribed_temperature = 10

# Apply the boundary conditions for prescribed temperatures
for node in prescribed_nodes:
    # Set corresponding row and column in K_global to zero
    K_global.row_del(node)
    K_global = K_global.row_insert(node, Matrix([[0] * num_nodes]))
    K_global.col_del(node)
    K_global = K_global.col_insert(node, Matrix([[0]] * num_nodes))
    
    # Set the diagonal element to 1
    K_global[node, node] = 1
    
    # Set the corresponding entry in the load vector to the prescribed temperature
    F[node] = prescribed_temperature

print("\nGlobal Stiffness Matrix K_global after applying boundary conditions:")
print(K_global)
print("\nLoad Vector F after applying boundary conditions:")
print(F)

# -----------------------------
# Solving the System
# -----------------------------

# Solve for nodal temperatures
T = K_global.LUsolve(F)

# Print the nodal temperatures
print("\nNodal Temperatures:")
print(T)

# -----------------------------
# End of Code for Heat Flux and Temperature at the Centroid
# -----------------------------

import sympy as sp

# Define the nodal temperatures (using the values from the solved T array)
T_nodal = sp.Matrix([10, 10, T[2], T[3], T[4]])

# -----------------------------
# For Element 1 (Rectangular Element)
# -----------------------------

# Define center coordinates for rectangular element (xi = 0, eta = 0)
xi_value = 0
eta_value = 0

# Substitute center coordinates into the B matrix
B_centroid = B.subs({xi: xi_value, eta: eta_value})

# Use nodal temperatures of Element 1 (nodes 1, 2, 3, 5)
T_element1 = sp.Matrix([T_nodal[0], T_nodal[1], T_nodal[2], T_nodal[4]])

# Calculate temperature at the centroid of Element 1 using shape functions
N_centroid = N.subs({xi: xi_value, eta: eta_value})
T_center_element1 = N_centroid.T * T_element1

print("\nTemperature at the centroid of Element 1:")
print(T_center_element1)

# Calculate heat flux at the centroid of Element 1
q_element1_centroid = -k_1 * B_centroid * T_element1

print("\nHeat Flux at the centroid of Element 1 (in global coordinates):")
print(q_element1_centroid)

# -----------------------------
# For Element 2 (Triangular Element)
# -----------------------------

# Define coordinates for triangular element (Element 2)
E2 = sp.Matrix([[2, 2],  # Node 3
                [2, 4],  # Node 4
                [0, 2]]) # Node 5

# Define shape functions for a three-node triangular element
# L1 = 1 - xi - eta, L2 = xi, L3 = eta
N_tri1 = 1 - xi - eta
N_tri2 = xi
N_tri3 = eta
N_tri = sp.Matrix([N_tri1, N_tri2, N_tri3])

# Calculate partial derivatives of triangular shape functions
dN_tri_dxi = [sp.diff(Ni, xi) for Ni in N_tri]
dN_tri_deta = [sp.diff(Ni, eta) for Ni in N_tri]

# Define the gradient matrix G_tri for triangular element
G_tri = sp.Matrix([dN_tri_dxi, dN_tri_deta])

# Multiply G_tri and E2 to get the Jacobian matrix for Element 2
J_tri = G_tri * E2
det_J_tri = J_tri.det()
J_inv_tri = J_tri.inv()

# Calculate B matrix for triangular element in global coordinates
B_tri = J_inv_tri * G_tri

# Centroid of a triangular element in natural coordinates
xi_value = 1/3
eta_value = 1/3

# Substitute centroid coordinates into the B matrix for Element 2
B_centroid_tri = B_tri.subs({xi: xi_value, eta: eta_value})

# Use nodal temperatures of Element 2 (nodes 3, 4, 5)
T_element2 = sp.Matrix([T_nodal[2], T_nodal[3], T_nodal[4]])

# Calculate temperature at the centroid of Element 2 using shape functions
N_centroid_tri = N_tri.subs({xi: xi_value, eta: eta_value})
T_center_element2 = N_centroid_tri.T * T_element2

print("\nTemperature at the centroid of Element 2:")
print(T_center_element2)

# Calculate heat flux at the centroid of Element 2
k_2 = 2  # Thermal conductivity for Element 2
q_element2_centroid = -k_2 * B_centroid_tri * T_element2

print("\nHeat Flux at the centroid of Element 2 (in global coordinates):")
print(q_element2_centroid)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

# Coordinates of nodes
nodes = np.array([
    [0, 0],  # Node 1
    [2, 0],  # Node 2
    [2, 2],  # Node 3
    [0, 2],  # Node 5
    [2, 4]   # Node 4 (additional node for the triangular element)
])

# Nodal temperatures
T_nodal = np.array([10, 10, float(T[2]), float(T[3]), float(T[4])])

# Create a Delaunay triangulation for the nodes
triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1])

# Create a contour plot using the nodal temperatures
plt.figure(figsize=(8, 6))
plt.tricontourf(triangulation, T_nodal, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.scatter(nodes[:, 0], nodes[:, 1], color='black', marker='o')
plt.title('Temperature Distribution across the Mesh')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.show()



# Centroid positions
centroids = np.array([
    [1, 1],       # Centroid of Element 1 (rectangular)
    [4/3, 8/3]    # Centroid of Element 2 (triangular)
])

# Heat flux components at the centroids (convert to float for plotting)
q_element1 = [float(q_element1_centroid[0]), float(q_element1_centroid[1])]
q_element2 = [float(q_element2_centroid[0]), float(q_element2_centroid[1])]

flux_vectors = np.array([q_element1, q_element2])

# Plot heat flux vectors
plt.figure(figsize=(8, 6))
plt.quiver(centroids[:, 0], centroids[:, 1], flux_vectors[:, 0], flux_vectors[:, 1],
           angles='xy', scale_units='xy', scale=1, color='blue')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='o')  # Mark centroids
plt.title('Heat Flux Vectors at Element Centroids')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.grid()
plt.axis('equal')
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Create the 3D plot of the temperature field
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Coordinates of the nodes
X = nodes[:, 0]
Y = nodes[:, 1]
Z = T_nodal

# Scatter plot of nodal temperatures
ax.scatter(X, Y, Z, c='r', marker='o', s=100)

# Add a surface plot interpolating the temperature
ax.plot_trisurf(X, Y, Z, cmap='hot', linewidth=0.2)

ax.set_title('3D Temperature Distribution')
ax.set_xlabel('X Coordinate (m)')
ax.set_ylabel('Y Coordinate (m)')
ax.set_zlabel('Temperature (°C)')

plt.show()


plt.figure(figsize=(8, 6))
plt.tricontour(nodes[:, 0], nodes[:, 1], [0, 1, 2, 4, 3], T_nodal, levels=10, linewidths=0.5, colors='k')
plt.tricontourf(nodes[:, 0], nodes[:, 1], [0, 1, 2, 4, 3], T_nodal, levels=10, cmap="RdBu_r")
plt.colorbar(label='Temperature (°C)')
plt.title('Contour Plot of Temperature')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.show()


