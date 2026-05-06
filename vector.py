class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x:.6f}, {self.y:.6f})"

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
        return (self.x, self.y)

    def norm(self):
        return (self.x**2 + self.y**2)**0.5

    def dot(self, other):
        return self.x * other.x + self.y * other.y