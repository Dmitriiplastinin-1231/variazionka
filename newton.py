import numpy as np
from vector import Vector
from armijo import armijo_line_search


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
    x = start
    hist = [x] if history else None

    for _ in range(max_iter):
        g = grad(x)
        if g.norm() < tol:
            break

        H = hessian(x)
        try:
            d_vec = np.linalg.solve(H, -np.array([g.x, g.y]))
            d = Vector(d_vec[0], d_vec[1])
        except np.linalg.LinAlgError:
            d = -g

        if use_line_search:
            alpha = armijo_line_search(f, x, d, g)

            if alpha == 0.0:
                d = -g
                alpha = armijo_line_search(f, x, d, g)

                if alpha == 0.0:
                    alpha = 1e-3
        else:
            alpha = 1.0

        x = x + alpha * d
        if history:
            hist.append(x)

    return x, f(x), hist