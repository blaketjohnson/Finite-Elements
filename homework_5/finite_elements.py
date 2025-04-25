import sympy as sp
import numpy as np

# Define shape functions for a quadrilateral parent element
def shape_functions_quad():
    xi, eta = sp.symbols('xi eta')
    N1 = (1/4) * (1 - xi) * (1 - eta)
    N2 = (1/4) * (1 + xi) * (1 - eta)
    N3 = (1/4) * (1 + xi) * (1 + eta)
    N4 = (1/4) * (1 - xi) * (1 + eta)
    return sp.Matrix([N1, N2, N3, N4])

# Define shape functions for a triangular parent element
def shape_functions_tri():
    xi, eta = sp.symbols('xi eta')
    N1 = 1 - xi - eta
    N2 = xi
    N3 = eta
    return sp.Matrix([N1, N2, N3])

# Simplify shape functions based on constant boundary flux
def simplify_shape_functions(shape_funcs, fixed_var, value):
    simplified_funcs = shape_funcs.subs(fixed_var, value)
    return simplified_funcs

# Compute Jacobian and its properties
def compute_jacobian(N, node_coordinates):
    # Define the symbols for parent coordinates
    xi, eta = sp.symbols('xi eta')

    # Compute the gradient of N with respect to xi and eta
    dN_dxi = [sp.diff(N[i], xi) for i in range(N.shape[0])]
    dN_deta = [sp.diff(N[i], eta) for i in range(N.shape[0])]

    # Create a matrix for the gradient of N (2 x num_nodes)
    Grad_N = sp.Matrix([[dN_dxi[i], dN_deta[i]] for i in range(N.shape[0])]).T
    print("Gradient of N with respect to xi and eta:")
    sp.pprint(Grad_N)

    # Create the coordinate matrix from node coordinates (num_nodes x 2)
    coord_matrix = sp.Matrix(node_coordinates)
    print("\nCoordinate Matrix (Global Coordinates):")
    sp.pprint(coord_matrix)

    # Calculate the Jacobian matrix by multiplying the gradient with the coordinate matrix
    J = Grad_N * coord_matrix
    print("\nJacobian Matrix (J):")
    sp.pprint(J)

    # Transpose of the Jacobian
    J_T = J.T
    print("\nTranspose of the Jacobian (J^T):")
    sp.pprint(J_T)

    # Calculate the Jacobian determinant and the Jacobian inverse if possible
    det_J = J.det().evalf(5)
    print("\nJacobian Determinant (|J|):")
    sp.pprint(det_J)
    
    if abs(det_J) > 1e-10:  # Check if determinant is non-zero
        J_inv = J.inv().evalf(5)
        print("\nJacobian Inverse (J^-1):")
        sp.pprint(J_inv)
    else:
        J_inv = None
        print("\nJacobian determinant is zero or near zero, cannot compute inverse.")

    return Grad_N, J, J_T, J_inv, det_J

# Compute B matrix
def compute_B_matrix(J_inv, Grad_N):
    # Calculate B matrix by multiplying the inverse of the Jacobian with the gradient of N
    if J_inv is None:
        print("\nJacobian inverse is not available, cannot compute B matrix.")
        return None
    
    B = J_inv * Grad_N
    print("\nB Matrix:")
    sp.pprint(B)
    return B

# Compute conduction matrix K using matrix B and matrix D
# Matrix D typically represents material properties such as elasticity or thermal conductivity matrix
def compute_conduction_matrix_with_D(B, D, element_type):
    if B is None or D is None:
        print("\nB matrix or D matrix is not available, cannot compute conduction matrix K.")
        return None
    
    # Define symbols for parent coordinates
    xi, eta = sp.symbols('xi eta')

    # Conduction matrix K before integration
    K_expr = B.T * D * B
    print("\nConduction Matrix Expression (before integration, K):")
    sp.pprint(K_expr)

    # Integrate over the parent element domain
    if element_type == 'quad':
        # Integration bounds for quadrilateral element: xi, eta in [-1, 1]
        K = sp.integrate(sp.integrate(K_expr, (xi, -1, 1)), (eta, -1, 1))
    elif element_type == 'tri':
        # Integration bounds for triangular element: xi, eta in [0, 1] with xi + eta <= 1
        K = sp.integrate(sp.integrate(K_expr, (eta, 0, 1 - xi)), (xi, 0, 1))
    else:
        raise ValueError("Invalid element type for integration.")
    
    print("\nConduction Matrix (K) after integration:")
    sp.pprint(K)
    return K

# Compute conduction matrix K with integration over parent element
def compute_conduction_matrix(B, det_J, k, element_type):
    if B is None or abs(det_J) < 1e-10:
        print("\nB matrix or Jacobian determinant is not available, cannot compute conduction matrix K.")
        return None
    
    # Define symbols for parent coordinates
    xi, eta = sp.symbols('xi eta')

    # Conduction matrix K before integration
    K_expr = k * B.T * B * abs(det_J)
    print("\nConduction Matrix Expression (before integration, K):")
    sp.pprint(K_expr)

    # Integrate over the parent element domain
    if element_type == 'quad':
        # Integration bounds for quadrilateral element: xi, eta in [-1, 1]
        K = sp.integrate(sp.integrate(K_expr, (xi, -1, 1)), (eta, -1, 1))
    elif element_type == 'tri':
        # Integration bounds for triangular element: xi, eta in [0, 1] with xi + eta <= 1
        K = sp.integrate(sp.integrate(K_expr, (eta, 0, 1 - xi)), (xi, 0, 1))
    else:
        raise ValueError("Invalid element type for integration.")
    
    print("\nConduction Matrix (K) after integration:")
    sp.pprint(K)
    return K

# Compute source matrix F
# Modified to match textbook approach using full shape functions
def compute_source_matrix(full_shape_funcs, q, det_J):
    # Define symbols for parent coordinates
    xi, eta = sp.symbols('xi eta')

    # Source matrix F using full shape functions and Jacobian determinant
    F_expr = q * full_shape_funcs * abs(det_J)
    print("\nSource Matrix Expression (before integration, F):")
    sp.pprint(F_expr)

    # Integrate over the parent element domain for quadrilateral: xi, eta in [-1, 1]
    F = sp.integrate(sp.integrate(F_expr, (xi, -1, 1)), (eta, -1, 1))
    
    print("\nSource Matrix (F) after integration:")
    sp.pprint(F)
    return F

# Compute boundary flux matrix H using textbook approach
def compute_boundary_flux_matrix(full_shape_funcs, boundary_flux, boundary):
    # Define symbols for parent coordinates
    xi, eta = sp.symbols('xi eta')

    # Determine which variable is fixed and integrate accordingly
    try:
        if 'xi' in boundary:
            fixed_value = float(boundary.split('=')[1].strip())
            # Assuming outward flux is negative
            if fixed_value == 1 or fixed_value == -1:
                boundary_flux = -abs(boundary_flux)
            else:
                boundary_flux = abs(boundary_flux)
            H_expr = boundary_flux * full_shape_funcs.subs(xi, fixed_value)
            H = sp.integrate(H_expr, (eta, -1, 1))
        elif 'eta' in boundary:
            fixed_value = float(boundary.split('=')[1].strip())
            # Assuming outward flux is negative
            if fixed_value == 1 or fixed_value == -1:
                boundary_flux = -abs(boundary_flux)
            else:
                boundary_flux = abs(boundary_flux)
            H_expr = boundary_flux * full_shape_funcs.subs(eta, fixed_value)
            H = sp.integrate(H_expr, (xi, -1, 1))
        else:
            raise ValueError("Invalid boundary specified. Please use 'xi = value' or 'eta = value'.")
    except (IndexError, ValueError) as e:
        raise ValueError("Boundary input is incorrect. Please use the format 'xi = value' or 'eta = value'.") from e
    
    print("\nBoundary Flux Matrix (H) after integration:")
    sp.pprint(H)
    return H

# Main function
def finite_element_preprocessor(element_type, node_coordinates):
    # Map nodes to parent element coordinates
    if element_type == 'quad':
        # Parent coordinates for a quadrilateral
        parent_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        shape_funcs = shape_functions_quad()
    elif element_type == 'tri':
        # Parent coordinates for a triangle
        parent_coords = [(0, 0), (1, 0), (0, 1)]
        shape_funcs = shape_functions_tri()
    else:
        raise ValueError("Element type must be 'quad' or 'tri'")
    
    # Create the coordinate matrix
    coord_matrix = {}
    for i, coord in enumerate(node_coordinates):
        coord_matrix[f'Node {i+1}'] = {'Global': coord, 'Parent': parent_coords[i]}
    
    # Print coordinate matrix
    print("\nCoordinate Matrix:")
    for node, coords in coord_matrix.items():
        print(f"{node}: Global {coords['Global']}, Parent {coords['Parent']}")
    
    # Print shape functions
    print("\nShape Functions (N):")
    sp.pprint(shape_funcs)
    
    # Store full shape functions for later Jacobian calculation
    full_shape_funcs = shape_funcs

    # Compute Jacobian and its properties using the full shape functions
    Grad_N, J, J_T, J_inv, det_J = compute_jacobian(full_shape_funcs, node_coordinates)

    # Compute B matrix
    B = compute_B_matrix(J_inv, Grad_N)

    # Ask user if there is a constant boundary flux to simplify the shape functions
    response = input("\nIs there a constant boundary flux (yes/no)? ").strip().lower()
    if response == 'yes':
        fixed_var = input("Enter the variable to fix (xi or eta): ").strip()
        value = float(input(f"Enter the value for {fixed_var}: "))
        if fixed_var == 'xi':
            fixed_var = sp.Symbol('xi')
        elif fixed_var == 'eta':
            fixed_var = sp.Symbol('eta')
        else:
            raise ValueError("Invalid variable. Please enter 'xi' or 'eta'.")
        
        simplified_funcs = simplify_shape_functions(shape_funcs, fixed_var, value)
        print("\nSimplified Shape Functions (N):")
        sp.pprint(simplified_funcs)
        shape_funcs = simplified_funcs

    # Get thermal conductivity value and compute conduction matrix K
    k = float(input("\nEnter the thermal conductivity value (k): "))
    K = compute_conduction_matrix(B, det_J, k, element_type)

    # Get source term value and compute source matrix F
    q = float(input("\nEnter the source term value (q): "))
    F = compute_source_matrix(full_shape_funcs, q, det_J)

    # Get boundary flux value and compute boundary flux matrix H
    boundary_flux = float(input("\nEnter the boundary flux value: "))
    boundary = input("Enter the boundary (e.g., 'xi = -1', 'xi = 1', 'eta = -1', 'eta = 1'): ").strip()
    H = compute_boundary_flux_matrix(full_shape_funcs, boundary_flux, boundary)

# Example usage
#finite_element_preprocessor("quad", [(2, 0), (2, 1), (0, 1), (0, 0)])

























