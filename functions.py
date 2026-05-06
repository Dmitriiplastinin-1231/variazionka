import numpy as np
from vector import Vector

def f1(point):
    x, y = point.c()
    return 100 * (y - x**2)**2 + 5 * (1 - x)**2

def grad_f1(point):
    x, y = point.c()
    dx = -400 * x * (y - x**2) - 10 * (1 - x)
    dy = 200 * (y - x**2)
    return Vector(dx, dy)

def hessian_f1(point):
    x, y = point.c()
    H11 = -400 * (y - 3*x**2) + 800*x**2 + 10
    H12 = -400 * x
    H21 = H12
    H22 = 200.0
    return np.array([[H11, H12], [H21, H22]])

def f2(point):
    x, y = point.c()
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def grad_f2(point):
    x, y = point.c()
    dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return Vector(dx, dy)

def hessian_f2(point):
    x, y = point.c()
    H11 = 12*x**2 + 4*y - 42
    H12 = 4*x + 4*y
    H21 = H12
    H22 = 12*y**2 + 4*x - 26
    return np.array([[H11, H12], [H21, H22]])