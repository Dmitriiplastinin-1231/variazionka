class Vector:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        x_repr = f"{self.x:.6f}"
        y_repr = f"{self.y:.6f}"
        return f"({x_repr}, {y_repr})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __rmul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar)

    def c(self):
        return self.x, self.y

    def norm(self):
        return (self.x * self.x + self.y * self.y) ** 0.5

    def dot(self, other):
        return self.x * other.x + self.y * other.y
