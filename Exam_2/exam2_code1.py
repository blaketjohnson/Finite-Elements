import sympy as sp

# Define constants and variable
xi = sp.Symbol('xi')
L = sp.Symbol('L', real=True, positive=True)
A = sp.Symbol('A', real=True, positive=True)
E = sp.Symbol('E', real=True, positive=True)

# Define the components of the B^e matrix
B1 = (2 / L) * (xi - sp.Rational(1, 2))
B2 = -4 * xi / L
B3 = (2 / L) * (xi + sp.Rational(1, 2))

# Construct B^e matrix as a column vector
B_matrix = sp.Matrix([B1, B2, B3])

# Calculate the integrand matrix (B^e)^T * B^e
K_e_integrand = B_matrix * B_matrix.T  # This is (B^e)^T * B^e

# Integrate each entry in the stiffness matrix
K11 = sp.integrate(K_e_integrand[0, 0], (xi, -1, 1))
K12 = sp.integrate(K_e_integrand[0, 1], (xi, -1, 1))
K13 = sp.integrate(K_e_integrand[0, 2], (xi, -1, 1))
K22 = sp.integrate(K_e_integrand[1, 1], (xi, -1, 1))
K23 = sp.integrate(K_e_integrand[1, 2], (xi, -1, 1))
K33 = sp.integrate(K_e_integrand[2, 2], (xi, -1, 1))

# Assemble the stiffness matrix with AE/(3L) factored out
K_e_matrix = sp.Matrix([
    [K11, K12, K13],
    [K12, K22, K23],
    [K13, K23, K33]
])

# Manually simplify by factoring out AE/(3L)
K_e_factored = (A * E / (3 * L)) * K_e_matrix.applyfunc(lambda x: sp.simplify(3 * x))

# Display final result
print("Final element stiffness matrix K^e with constants factored out:")
sp.pprint(K_e_factored)

# Optionally save to output file
with open("output.txt", "w") as f:
    f.write(str(K_e_factored))


