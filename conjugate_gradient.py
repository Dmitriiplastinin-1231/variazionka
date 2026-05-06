from armijo import armijo_line_search

def conjugate_gradient_fr(f, grad, start, max_iter=1000, tol=1e-6, history=True):
    x = start
    g = grad(x)
    d = -g
    hist = [x] if history else None
    for _ in range(max_iter):
        if g.norm() < tol:
            break
        alpha = armijo_line_search(f, x, d, g)
        x_new = x + alpha * d
        g_new = grad(x_new)
        beta = g_new.dot(g_new) / g.dot(g)
        d = -g_new + beta * d
        x, g = x_new, g_new
        if history:
            hist.append(x)
    return x, f(x), hist