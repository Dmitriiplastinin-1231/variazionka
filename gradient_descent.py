from armijo import armijo_line_search


def gradient_descent(f, grad, start, max_iter=1000, tol=1e-6, history=True):
    current = start
    track = [current] if history else None

    for _ in range(max_iter):
        g = grad(current)
        if g.norm() < tol:
            break

        direction = -g
        step = armijo_line_search(f, current, direction, g)
        current = current + step * direction

        if history:
            track.append(current)

    return current, f(current), track
