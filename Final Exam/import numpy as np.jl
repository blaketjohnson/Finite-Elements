import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from IPython.core.display import HTML
from sympy import Matrix
import matplotlib.tri as tri

'''
For the first part of this problem I followed example 3.1 from the 
'''

"""
Lets set variables for the equation and known Boundaries
"""

# Define the variable and function
x = sp.symbols('x')
u = sp.Function('u')(x)

# Define the equation parameters
f = 0  # given as 0
ubar = 1  # boundary condition at x=0
qbar = 0  # boundary condition at x=1

# Differential equation
diff_eq = -sp.diff((1 + x) * sp.diff(u, x), x) - f

# Print the equation
print("Differential Equation:", diff_eq)

# Boundary conditions as variables
bc1 = u.subs(x, 0) - ubar  # u(0) = ubar
bc2 = ((1 + x) * sp.diff(u, x)).subs(x, 1) - qbar  # [(1+x) * du/dx]|_{x=1} = qbar

print("Boundary Condition at x=0:", bc1)
print("Boundary Condition at x=1:", bc2)

"""
Next step is to calculate the weak form of the equation
"""

# Define variables and functions
x = sp.Symbol('x')                  # Spatial variable
u = sp.Function('u')(x)             # Trial function
w = sp.Function('w')(x)             # Test function
f = sp.Function('f')(x)             # Source term, now a function of x

# Define the weak form
du_dx = sp.diff(u, x)               # Derivative of u
dw_dx = sp.diff(w, x)               # Derivative of w
weak_form = sp.integrate((1 + x) * du_dx * dw_dx, (x, 0, 1)) - sp.integrate(w * f, (x, 0, 1))

# Display the weak form
print("Weak form of the equation:")
sp.pretty_print(weak_form)

# Integration by parts to handle the first term
boundary_term = w * (1 + x) * du_dx    # Boundary term after integration by parts
boundary_contrib = boundary_term.subs(x, 1) - boundary_term.subs(x, 0)

# Reformulate the weak form with boundary contributions
weak_form = boundary_contrib - sp.integrate(dw_dx * (1 + x) * du_dx, (x, 0, 1))

# Simplify and display results
print("Boundary Contribution:")
sp.pretty_print(boundary_contrib)

print("\nWeak Form After Integration by Parts:")
sp.pretty_print(weak_form)

'''
The weight function is zero at x = 0 due to the Dirichlet boundary condition.
w(x) = 0 at x = 0.
So I simplified the weak form to get my final weak form.
'''

# Simplified weak form with boundary term at x=1
boundary_term_simplified = boundary_term.subs(x, 1)  # Only keep x=1 contribution
weak_form_simplified = boundary_term_simplified - sp.integrate(dw_dx * (1 + x) * du_dx, (x, 0, 1))

# Display results
print("Simplified Weak Form:")
sp.pretty_print(weak_form_simplified)

'''
Now that I have the weak form I followed chapter 4 to construct 
two linear element shape functions and the corresponding shape function

Since they are linear and I need two, I decided to use two of the same shape functions
element 1: [0,0.5]
element 2: [0.5,1]
I used the equations from pg 83 for the shape functions
'''


# Define the variable and nodes
x = sp.Symbol('x')
x1, x2, x3 = 0, 0.5, 1  # Nodes for the two-element mesh

# Local shape functions for Element 1 (0 to 0.5)
N1_e1 = (x - x2) / (x1 - x2)  # Shape function for Node 1 (updated convention)
N2_e1 = (x - x1) / (x2 - x1)  # Shape function for Node 2 (unchanged)

# Local shape functions for Element 2 (0.5 to 1)
N1_e2 = (x - x3) / (x2 - x3)  # Shape function for Node 2 (shared, updated convention)
N2_e2 = (x - x2) / (x3 - x2)  # Shape function for Node 3 (unchanged)

# Display the shape functions
print("Shape Functions for Element 1:")
sp.pretty_print(N1_e1)
sp.pretty_print(N2_e1)

print("\nShape Functions for Element 2:")
sp.pretty_print(N1_e2)
sp.pretty_print(N2_e2)

"""
Next will be to set up the global shape functions
I want to set up matrices to do the math
"""

# Define variables and element limits
x = sp.Symbol('x')  # Spatial variable
x1, x2 = sp.symbols('x1 x2')  # Node positions (e.g., 0 to 0.5 for Element 1)

# Define element shape functions
N1 = (x - x2) / (x1 - x2)  # Shape function for Node 1
N2 = (x - x1) / (x2 - x1)  # Shape function for Node 2

# Shape function derivatives
dN1_dx = sp.diff(N1, x)
dN2_dx = sp.diff(N2, x)

# Set up the B matrix, where B is the derivative of N with respect to x

# Shape function derivatives
dN1_dx = sp.diff(N1, x)
dN2_dx = sp.diff(N2, x)

# B-matrix (derivatives of shape functions)
B = sp.Matrix([dN1_dx, dN2_dx])

AkBeTB = (1+x)*B.T*B

print("B Matrix:")
sp.pprint(AkBeTB)
"""
Now using B I set up the elemental stiffness matrix, K
"""

# Stiffness matrix Ke for one element
Ke = sp.Matrix(2, 2, lambda i, j: sp.integrate(B[i] * (1 + x) * B[j], (x, x1, x2)))

# Simplify the generic stiffness matrix
Ke_simplified = sp.simplify(Ke)

# Display the simplified stiffness matrix
print("Simplified Element Stiffness Matrix (Ke):")
sp.pprint(Ke_simplified)

# Substitute values for Element 1: [0, 0.5]
Ke_e1 = Ke.subs({x1: 0, x2: 0.5})
print("\nElement Stiffness Matrix for Element 1 (Ke_e1):")
sp.pprint(Ke_e1)

# Substitute values for Element 2: [0.5, 1]
Ke_e2 = Ke.subs({x1: 0.5, x2: 1})
print("\nElement Stiffness Matrix for Element 2 (Ke_e2):")
sp.pprint(Ke_e2)

'''
Next we can consider the element external force matrix.
In this problem the external force is 0 so the matrix will be 0
'''



'''
Assemble the Global Stiffness Matrix
'''

# Set up a matrix of zeros for the global stiffness matrix
K_global = sp.zeros(3, 3)

# Add contributions from Element 1
K_global[0:2, 0:2] += Ke_e1

# Add contributions from Element 2
K_global[1:3, 1:3] += Ke_e2

print("\nGlobal Stiffness Matrix (K_global):")
sp.pprint(K_global)

# Apply Dirichlet boundary condition at x = 0 (u(0) = 1)
for j in range(K_global.shape[1]):  # Set row 0 to 0
    K_global[0, j] = 0
for i in range(K_global.shape[0]):  # Set column 0 to 0
    K_global[i, 0] = 0
K_global[0, 0] = 1  # Set the diagonal entry to 1

# Initialize global load vector (3x1)
F_global = sp.zeros(3, 1)

# Apply Dirichlet boundary condition to the load vector
F_global[0] = 1  # u(0) = 1

# Apply Neumann boundary condition at x = 1 (q = 0)
# (No changes needed to F_global since q = 0 contributes nothing)

# Solve for nodal displacements
u_global = sp.simplify(K_global.inv() * F_global)
print("\nNodal Displacements (u_global):")
sp.pprint(u_global)