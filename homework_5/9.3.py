'''
Problem 9.3

Consider a quadrilateral domain model of unit thickness with a single finite element as shown in Figure 9.15.
 All dimensions are in meters. The traction applied on the edge 1 - 2 is normal to the edge and is given by 
 $ 6 \cdot n N m^{-2}$, where n is the unit vector normal to the edge.
Calculate the element boundary force matrix.
'''
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
import sympy as sp

# Define given values
tau = 6  # Magnitude of traction (in N/m^2)
'''
Set up a matrix for the normal vector n for the edge 1-2
'''
# Edge vector from Node 1 to Node 2
edge_1_2 = sp.Matrix([0.6, 1.5])

# the norm of the edge vector
mag_edge = edge_1_2.norm()

# normalize the edge vector
norm_edge = edge_1_2 / edge_1_2.norm()

print("\nnormalized edge vector (norm_edge):")
sp.pprint(norm_edge)

n_x = - norm_edge[1] # x component of the normal vector in parent coordinates
n_y = norm_edge[0] # y component of the normal vector in parent coordinates
n = sp.Matrix([n_x, n_y]) # normal vector in parent coordinates

'''
Calculate the traction
'''
# t = tau * n_1_2
t = tau*n
t_x = t[0]
t_y = t[1]

print("\nTraction vector (t):")
sp.pprint(t)

print("\nTraction vector x component (t_x):")
sp.pprint(t_x)

print("\nTraction vector y component (t_y):")
sp.pprint(t_y)

'''
Shape Functions
'''
xi = sp.symbols('xi')  # Natural coordinate along the edge

N_1 = (1/2) * (1 - xi)  # Shape function for Node 1 on edge 1-2
N_2 = (1/2) * (1 + xi)  # Shape function for Node 2 on edge 1-2

N = sp.Matrix([N_1, N_2])

print("\nShape function for Node 1 (N_1):")
sp.pprint(N_1)

print("\nShape function for Node 2 (N_2):")
sp.pprint(N_2)

print("\nShape function matrix (N):")
sp.pprint(N)

'''
Boundary force vector
'''
# Compute force contributions in x-direction
f_x = sp.Matrix([
    sp.integrate(N_1 * t_x * mag_edge, (xi, -1, 1)),  # Node 1
    sp.integrate(N_2 * t_x * mag_edge, (xi, -1, 1))   # Node 2
])

# Compute force contributions in y-direction
f_y = sp.Matrix([
    sp.integrate(N_1 * t_y * mag_edge, (xi, -1, 1)),  # Node 1
    sp.integrate(N_2 * t_y * mag_edge, (xi, -1, 1))   # Node 2
])

# Assemble boundary force vector
f = sp.Matrix.vstack(f_x, f_y)

print("\nBoundary force vector (f):")
sp.pprint(f)



fx = sp.integrate(N_1 * t_x * (mag_edge), (xi, -1, 1)) + sp.integrate(N_2 * t_x * (mag_edge), (xi, -1, 1))
fy = sp.integrate(N_1 * t_y * (mag_edge), (xi, -1, 1)) + sp.integrate(N_2 * t_y * (mag_edge), (xi, -1, 1))

print("\nForce contribution at Node 1 (f1):")
sp.pprint(fx)

print("\nForce contribution at Node 2 (f2):")
sp.pprint(fy)





