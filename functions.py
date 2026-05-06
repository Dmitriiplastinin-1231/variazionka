import numpy as np

from vector import Vector


def f1(point):
    x, y = point.c()
    diff = y - x * x
    return 100 * diff * diff + 5 * (1 - x) * (1 - x)


def grad_f1(point):
    x, y = point.c()
    diff = y - x * x
    dx = -400 * x * diff - 10 * (1 - x)
    dy = 200 * diff
    return Vector(dx, dy)


def hessian_f1(point):
    x, y = point.c()
    h11 = -400 * (y - 3 * x * x) + 800 * x * x + 10
    h12 = -400 * x
    h21 = h12
    h22 = 200.0
    return np.array([[h11, h12], [h21, h22]])


def f2(point):
    x, y = point.c()
    term1 = x * x + y - 11
    term2 = x + y * y - 7
    return term1 * term1 + term2 * term2


def grad_f2(point):
    x, y = point.c()
    term1 = x * x + y - 11
    term2 = x + y * y - 7
    dx = 4 * x * term1 + 2 * term2
    dy = 2 * term1 + 4 * y * term2
    return Vector(dx, dy)


def hessian_f2(point):
    x, y = point.c()
    h11 = 12 * x * x + 4 * y - 42
    h12 = 4 * x + 4 * y
    h21 = h12
    h22 = 12 * y * y + 4 * x - 26
    return np.array([[h11, h12], [h21, h22]])
