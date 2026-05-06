def armijo_line_search(
    f,
    x,
    d,
    grad_x,
    alpha0=1.0,
    c=1e-4,
    beta=0.5,
    min_alpha=1e-12,
    max_iter=60,
):
    f_at_x = f(x)
    slope = grad_x.dot(d)

    if slope >= 0:
        return 0.0

    step = alpha0
    for _ in range(max_iter):
        if step < min_alpha:
            return 0.0

        if f(x + step * d) <= f_at_x + c * step * slope:
            return step

        step *= beta

    return 0.0
