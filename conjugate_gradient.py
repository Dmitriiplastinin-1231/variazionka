from armijo import armijo_line_search


def conjugate_gradient_fr(f, grad, start, max_iter=1000, tol=1e-6, history=True):
    point = start
    grad_val = grad(point)
    direction = -grad_val
    track = [point] if history else None

    for _ in range(max_iter):
        if grad_val.norm() < tol:
            break

        step = armijo_line_search(f, point, direction, grad_val)
        next_point = point + step * direction
        next_grad = grad(next_point)

        beta = next_grad.dot(next_grad) / grad_val.dot(grad_val)
        direction = -next_grad + beta * direction

        point, grad_val = next_point, next_grad
        if history:
            track.append(point)

    return point, f(point), track
