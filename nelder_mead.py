from vector import Vector

def nelder_mead(f, start, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                max_iter=1000, tol=1e-6, history=True):
    v1 = start
    v2 = Vector(start.x + 1.0, start.y)
    v3 = Vector(start.x, start.y + 1.0)
    simplex = [v1, v2, v3]
    hist = [] if history else None

    for _ in range(max_iter):
        values = [(v, f(v)) for v in simplex]
        values.sort(key=lambda x: x[1])
        b, g, w = values[0][0], values[1][0], values[2][0]
        f_b, f_g, f_w = values[0][1], values[1][1], values[2][1]

        if history:
            hist.append(b)

        if max(f_b, f_g, f_w) - min(f_b, f_g, f_w) < tol:
            break

        centroid = (b + g) / 2.0
        xr = centroid + alpha * (centroid - w)
        f_xr = f(xr)

        if f_xr < f_g:
            if f_xr < f_b:
                xe = centroid + gamma * (xr - centroid)
                if f(xe) < f_xr:
                    simplex[2] = xe
                else:
                    simplex[2] = xr
            else:
                simplex[2] = xr
        else:
            if f_xr < f_w:
                simplex[2] = xr
            xc = centroid + rho * (w - centroid)
            if f(xc) < f_w:
                simplex[2] = xc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = b + sigma * (simplex[i] - b)

        simplex = [b, g, simplex[2]]

    best = min(simplex, key=f)
    return best, f(best), hist