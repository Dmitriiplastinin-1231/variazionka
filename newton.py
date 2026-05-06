import numpy as np

from armijo import armijo_line_search
from vector import Vector


def newton_method(
    f,
    grad,
    hessian,
    start,
    max_iter=1000,
    tol=1e-6,
    use_line_search=True,
    history=True,
):
    point = start
    track = [point] if history else None

    for _ in range(max_iter):
        g = grad(point)
        if g.norm() < tol:
            break

        hess = hessian(point)
        try:
            step_vec = np.linalg.solve(hess, -np.array([g.x, g.y]))
            direction = Vector(step_vec[0], step_vec[1])
        except np.linalg.LinAlgError:
            direction = -g

        if use_line_search:
            alpha = armijo_line_search(f, point, direction, g)
            if alpha == 0.0:
                direction = -g
                alpha = armijo_line_search(f, point, direction, g)
                if alpha == 0.0:
                    alpha = 1e-3
        else:
            alpha = 1.0

        point = point + alpha * direction
        if history:
            track.append(point)

    return point, f(point), track
