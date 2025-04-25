def compute_B_row1():

    # Define symbolic variables for local coordinates
    xi, eta = sp.symbols('xi eta')

    # Define the quadratic shape functions for the six-node triangle
    N1 = xi * (2 * xi - 1)
    N2 = eta * (2 * eta - 1)
    N3 = (1 - xi - eta) * (2 * (1 - xi - eta) - 1)
    N4 = 4 * xi * eta
    N5 = 4 * eta * (1 - xi - eta)
    N6 = 4 * xi * (1 - xi - eta)

    # List of shape functions
    shape_functions = [N1, N2, N3, N4, N5, N6]

    # Assume nodal coordinates for simplicity
    nodal_coords = {
        'x1': 0, 'y1': 0,
        'x2': 1, 'y2': 0,
        'x3': 0, 'y3': 1,
        'x4': 0.5, 'y4': 0,
        'x5': 0.5, 'y5': 0.5,
        'x6': 0, 'y6': 0.5
    }

    # Map global coordinates (substitute numerical values for simplicity)
    x_global = sum(Ni * nodal_coords[f'x{i+1}'] for i, Ni in enumerate(shape_functions))
    y_global = sum(Ni * nodal_coords[f'y{i+1}'] for i, Ni in enumerate(shape_functions))

    # Compute the Jacobian matrix
    J = sp.Matrix([
        [sp.diff(x_global, xi), sp.diff(x_global, eta)],
        [sp.diff(y_global, xi), sp.diff(y_global, eta)]
    ])

    # Compute the Jacobian determinant and inverse
    J_det = J.det()
    if J_det != 0:
        J_inv = J.inv()

        # Transform derivatives of shape functions from (xi, eta) to (x, y)
        dN_dx = [J_inv[0, 0] * sp.diff(N, xi) + J_inv[0, 1] * sp.diff(N, eta) for N in shape_functions]
        dN_dy = [J_inv[1, 0] * sp.diff(N, xi) + J_inv[1, 1] * sp.diff(N, eta) for N in shape_functions]

        # Construct the first row of the B matrix
        B_row1 = []
        for i in range(len(dN_dx)):
            B_row1.append(dN_dx[i])
            B_row1.append(0)  # Zero placeholders for the uy components

        return sp.Matrix([B_row1])  # Return as a matrix for clean display
    else:
        raise ValueError("Jacobian is singular; check the mapping.")
    
# Part 1: Compute and display Row 1 of the B-matrix
B_row1 = compute_B_row1()
print("Row 1 of the B-matrix (epsilon_xx):")
print(B_row1)

# Part 2: Verify rigid body translation and display the result

def verify_rigid_body_translation(B_row1):
    import sympy as sp

    # Define the constant rigid body translation displacement field
    cx = sp.Symbol('c_x')  # Constant translation in x-direction
    cy = sp.Symbol('c_y')  # Constant translation in y-direction

    # Nodal displacement vector for rigid body translation
    nodal_displacements = [cx, cy] * (B_row1.shape[1] // 2)

    # Compute epsilon_xx using the first row of the B matrix and the displacement vector
    epsilon_xx = sum(B_row1[0, i] * nodal_displacements[i] for i in range(B_row1.shape[1]))

    # Simplify the result to check if epsilon_xx vanishes
    epsilon_xx_simplified = sp.simplify(epsilon_xx)

    print("Verification for rigid body translation:")
    print(f"ε_xx = {epsilon_xx_simplified}")



def compute_strain_field(B_row1):
    '''
    Computes the strain field (ε_xx) for the displacement field u_x = a*x and u_y = b*y.
    '''
    # Proportional displacements: u_x = a*x and u_y = b*y
    # These represent a uniform strain field induced by scaling in x and y directions.
    a, b = sp.symbols('a b')  # Constants of proportionality

    # Define nodal coordinates (physical domain for the triangle)
    nodal_coords = {
        'x1': 0, 'y1': 0,
        'x2': 1, 'y2': 0,
        'x3': 0, 'y3': 1,
        'x4': 0.5, 'y4': 0,
        'x5': 0.5, 'y5': 0.5,
        'x6': 0, 'y6': 0.5
    }

    # Compute nodal displacements proportional to coordinates
    # Each node's displacement depends linearly on its x and y coordinates
    nodal_displacements = []
    for i in range(1, 7):  # 6 nodes
        x_i = nodal_coords[f'x{i}']
        y_i = nodal_coords[f'y{i}']
        nodal_displacements.append(a * x_i)  # u_x = a * x
        nodal_displacements.append(b * y_i)  # u_y = b * y

    # Compute the strain field (ε_xx) using the first row of the B matrix
    epsilon_xx = sum(B_row1[0, i] * nodal_displacements[i] for i in range(B_row1.shape[1]))

    # Simplify the strain field
    epsilon_xx_simplified = sp.simplify(epsilon_xx)

    # Why it makes sense:
    # ε_xx = -1.0 * a corresponds to uniform strain along x-axis.
    # It depends only on a because u_x scales linearly with x.

    return epsilon_xx_simplified

def compute_additional_strains(B_matrix):
    '''
    Computes additional strain components (ε_yy and γ_xy) for the displacement field u_x = a*x, u_y = b*y.
    '''
    a, b = sp.symbols('a b')  # Constants for proportional displacement

    # Define nodal coordinates (as before)
    nodal_coords = {
        'x1': 0, 'y1': 0,
        'x2': 1, 'y2': 0,
        'x3': 0, 'y3': 1,
        'x4': 0.5, 'y4': 0,
        'x5': 0.5, 'y5': 0.5,
        'x6': 0, 'y6': 0.5
    }

    # Compute nodal displacements (as before)
    nodal_displacements = []
    for i in range(1, 7):  # 6 nodes
        x_i = nodal_coords[f'x{i}']
        y_i = nodal_coords[f'y{i}']
        nodal_displacements.append(a * x_i)  # u_x = a * x
        nodal_displacements.append(b * y_i)  # u_y = b * y

    # Compute epsilon_yy (Row 2 of B-matrix)
    epsilon_yy = sum(B_matrix[1, i] * nodal_displacements[i] for i in range(B_matrix.shape[1]))
    epsilon_yy_simplified = sp.simplify(epsilon_yy)

    # Compute gamma_xy (Row 3 of B-matrix)
    gamma_xy = sum(B_matrix[2, i] * nodal_displacements[i] for i in range(B_matrix.shape[1]))
    gamma_xy_simplified = sp.simplify(gamma_xy)

    # Why it makes sense:
    # ε_yy = 0 because there is no displacement change along y for u_y = b*y.
    # γ_xy = 0 because there is no shear displacement induced.

    return epsilon_yy_simplified, gamma_xy_simplified

# Main script
if __name__ == "__main__":
    # Compute Row 1 of the B-matrix
    B_row1 = compute_B_row1()
    print("Row 1 of the B-matrix (ε_xx):")
    print(B_row1)

    # Compute strain field ε_xx (Part b)
    epsilon_xx = compute_strain_field(B_row1)
    print("\nStrain field (ε_xx) for u_x = a*x, u_y = b*y:")
    print(epsilon_xx)

    # Extend B-matrix to include ε_yy and γ_xy (assume placeholder rows)
    B_full_matrix = sp.zeros(3, B_row1.shape[1])
    B_full_matrix[0, :] = B_row1  # Assign Row 1

    # Compute additional strain components
    epsilon_yy, gamma_xy = compute_additional_strains(B_full_matrix)
    print("\nStrain field (ε_yy) for u_x = a*x, u_y = b*y:")
    print(epsilon_yy)
    print("\nShear strain field (γ_xy) for u_x = a*x, u_y = b*y:")
    print(gamma_xy)