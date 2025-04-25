import sympy as sp

# Define variables and functions
x, y = sp.symbols('x y')              # Spatial variables
u = sp.Function('u')(x, y)            # Trial function
w = sp.Function('w')(x, y)            # Test function (weight function)
a1 = sp.Function('a1')(x, y)          # Coefficient a1
a2 = sp.Function('a2')(x, y)          # Coefficient a2
a0 = sp.Function('a0')(x, y)          # Coefficient a0
f = sp.Function('f')(x, y)            # Source term

# Differential equation form
diff_eq = -sp.diff(a1 * sp.diff(u, x), x) - sp.diff(a2 * sp.diff(u, y), y) + a0 * u - f

# Print the strong form of the equation
print("Differential Equation (Strong Form):")
sp.pretty_print(diff_eq)

"""
Derive the weak form by multiplying by the weight function and integrating over the domain R.
"""

# Define partial derivatives for weak form
du_dx = sp.diff(u, x)
du_dy = sp.diff(u, y)
dw_dx = sp.diff(w, x)
dw_dy = sp.diff(w, y)

# Set up the weak form by multiplying the PDE by the test function and integrating over the domain
# For simplicity, we represent the domain as R without explicitly specifying the limits
weak_form = (w * diff_eq).simplify()

# Apply integration by parts to reduce the order of derivatives in the weak form
# Integration by parts for the x-direction
integrate_x_part = sp.integrate(-w * sp.diff(a1 * du_dx, x), (x, 0, 1))
boundary_term_x = (w * a1 * du_dx).subs(x, 1) - (w * a1 * du_dx).subs(x, 0)

# Integration by parts for the y-direction
integrate_y_part = sp.integrate(-w * sp.diff(a2 * du_dy, y), (y, 0, 1))
boundary_term_y = (w * a2 * du_dy).subs(y, 1) - (w * a2 * du_dy).subs(y, 0)

# Combine contributions for weak form
weak_form_combined = (integrate_x_part + integrate_y_part + a0 * u * w).simplify()

# Reformulate the weak form including boundary contributions
weak_form_with_boundary = (boundary_term_x + boundary_term_y + weak_form_combined).simplify()

print("\nWeak Form before Applying Boundary Conditions:")
sp.pretty_print(weak_form_with_boundary)

"""
Simplify the weak form by applying boundary conditions.
The weight function is zero on Dirichlet boundary (S1).
"""

# Assume w = 0 on S1 (Dirichlet boundary)
boundary_term_simplified = (boundary_term_x + boundary_term_y).subs(w, 0)

# Final weak form after simplification
weak_form_simplified = weak_form_combined + boundary_term_simplified

print("\nSimplified Weak Form:")
sp.pretty_print(weak_form_simplified)





