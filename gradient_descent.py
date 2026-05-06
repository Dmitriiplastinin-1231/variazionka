from armijo import armijo_line_search

def gradient_descent(f, grad, start, max_iter=1000, tol=1e-6, history=True):
    x = start
    hist = [x] if history else None
    for _ in range(max_iter):
        g = grad(x)
        if g.norm() < tol:
            break
        d = -g
        alpha = armijo_line_search(f, x, d, g)
        x = x + alpha * d
        if history:
            hist.append(x)
    return x, f(x), hist

