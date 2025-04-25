% File: plot_open_bspline_spline_only.m
% Description: Plots an open uniform cubic B-spline curve (spline only)
% using a clamped knot vector and the Cox-de Boor recursion formula.

function plot_open_bspline_spline_only()
    % Define the 6 control points (each row is [x, y])
    ctrl_pts = [ 0,  0;   % p0
                -1,  1;   % p1
                -1,  3;   % p2
                 0,  4;   % p3
                 0,  6;   % p4
                -3,  8];  % p5

    % For a cubic B-spline (degree = 3, order = 4) with 6 control points:
    % The knot vector length is n + degree + 2, where n = 5.
    % A typical clamped (open) knot vector (normalized) is:
    knot = [0, 0, 0, 0, 1/3, 2/3, 1, 1, 1, 1];
    
    % The valid parameter domain is u in [knot(4), knot(7)] = [0, 1]
    u_vals = linspace(0, 1, 200);
    curve = zeros(length(u_vals), 2);
    
    % Evaluate the B-spline curve at each u value.
    for j = 1:length(u_vals)
        u = u_vals(j);
        curve(j, :) = bspline_point(u, ctrl_pts, knot, 4);
    end

    % Plot the B-spline curve (without connecting the endpoints)
    figure;
    plot(curve(:,1), curve(:,2), 'b-', 'LineWidth', 2, 'DisplayName', 'B-spline Curve');
    title('Classic Open Uniform Cubic B-Spline Curve (Spline Only)');
    xlabel('x');
    ylabel('y');
    grid on;
    axis equal;
    legend('Location', 'best');
end

function pt = bspline_point(u, ctrl_pts, knot, order)
    % Evaluate the B-spline curve point at parameter u using Cox-de Boor recursion.
    n = size(ctrl_pts, 1) - 1;
    pt = [0, 0];
    for i = 0:n
        pt = pt + cox_de_boor(u, i, order, knot) * ctrl_pts(i+1, :);
    end
end

function N = cox_de_boor(u, i, k, knot)
    % Recursively evaluate the i-th B-spline basis function of order k at u.
    % u: parameter value
    % i: index (0-indexed)
    % k: order (degree + 1)
    % knot: knot vector
    if k == 1
        % Zero-degree basis function: 1 if u is in [knot(i+1), knot(i+2))
        if (knot(i+1) <= u && u < knot(i+2)) || (u == knot(end) && i == length(knot)-2)
            N = 1;
        else
            N = 0;
        end
    else
        denom1 = knot(i+k) - knot(i+1);
        if denom1 == 0
            term1 = 0;
        else
            term1 = (u - knot(i+1)) / denom1 * cox_de_boor(u, i, k-1, knot);
        end
        
        denom2 = knot(i+k+1) - knot(i+2);
        if denom2 == 0
            term2 = 0;
        else
            term2 = (knot(i+k+1) - u) / denom2 * cox_de_boor(u, i+1, k-1, knot);
        end
        N = term1 + term2;
    end
end
