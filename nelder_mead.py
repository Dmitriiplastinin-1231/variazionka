from vector import Vector


def nelder_mead(
    f,
    start,
    alpha=1.0,
    gamma=2.0,
    rho=0.5,
    sigma=0.5,
    max_iter=1000,
    tol=1e-6,
    history=True,
):
    v1 = start
    v2 = Vector(start.x + 1.0, start.y)
    v3 = Vector(start.x, start.y + 1.0)
    simplex = [v1, v2, v3]
    track = [] if history else None

    for _ in range(max_iter):
        ranked = [(vertex, f(vertex)) for vertex in simplex]
        ranked.sort(key=lambda item: item[1])

        best, good, worst = ranked[0][0], ranked[1][0], ranked[2][0]
        f_best, f_good, f_worst = ranked[0][1], ranked[1][1], ranked[2][1]

        if history:
            track.append(best)

        if max(f_best, f_good, f_worst) - min(f_best, f_good, f_worst) < tol:
            break

        centroid = (best + good) / 2.0
        reflected = centroid + alpha * (centroid - worst)
        f_reflected = f(reflected)

        if f_reflected < f_good:
            if f_reflected < f_best:
                expanded = centroid + gamma * (reflected - centroid)
                simplex[2] = expanded if f(expanded) < f_reflected else reflected
            else:
                simplex[2] = reflected
        else:
            if f_reflected < f_worst:
                simplex[2] = reflected
            contracted = centroid + rho * (worst - centroid)
            if f(contracted) < f_worst:
                simplex[2] = contracted
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = best + sigma * (simplex[i] - best)

        simplex = [best, good, simplex[2]]

    best_vertex = min(simplex, key=f)
    return best_vertex, f(best_vertex), track
