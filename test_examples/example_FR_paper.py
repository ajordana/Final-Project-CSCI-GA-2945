import numpy as np
import sys


class g1:
    def __init__(self):
        self.f = 0
        self.fx = 0
        self.fy = 0
        self.fxx = 0
        self.fyy = 0
        self.fyx = 0

    def calc(self, x, y):
        self.f = self.eval(x, y)
        self.fx = -6 * x + 4 * y
        self.fy = -2 * y + 4 * x
        self.fxx = -6
        self.fyy = -2
        self.fyx = 4
        self.fxy = self.fyx

    def eval(self, x, y):
        return -3 * x ** 2 - y ** 2 + 4 * x * y


class g2:
    def __init__(self):
        self.f = 0
        self.fx = 0
        self.fy = 0
        self.fxx = 0
        self.fyy = 0
        self.fyx = 0

    def calc(self, x, y):
        self.f = self.eval(x, y)
        self.fx = 6 * x + 4 * y
        self.fy = 2 * y + 4 * x
        self.fxx = 6
        self.fyy = 2
        self.fyx = 4
        self.fxy = self.fyx

    def eval(self, x, y):
        return 3 * x ** 2 + y ** 2 + 4 * x * y


class g3:
    def __init__(self):
        self.f = 0
        self.fx = 0
        self.fy = 0
        self.fxx = 0
        self.fyy = 0
        self.fyx = 0

    def calc(self, x, y):
        pol = 4 * x ** 2 - (y - 3 * x + 0.05 * x ** 3) ** 2 - 0.1 * y ** 4
        self.f = pol * self.exp(x, y)

        polx = 8 * x - 2 * (-3 + 0.05 * 3 * x ** 2) * (y - 3 * x + 0.05 * x ** 3)
        self.fx = polx * self.exp(x, y) - 0.02 * x * pol * self.exp(x, y)
        self.fyx = (
            -2 * (-3 + 0.05 * 3 * x ** 2) * self.exp(x, y)
            - 0.02 * y * polx * self.exp(x, y)
            + 0.02 * x * 0.02 * y * pol * self.exp(x, y)
        )
        self.fxy = self.fyx
        self.fxx = "not implemented"

        poly = -2 * (y - 3 * x + 0.05 * x ** 3) - 0.4 * y ** 3
        self.fy = poly * self.exp(x, y) - 0.02 * y * pol * self.exp(x, y)
        self.fyy = (
            (-2 - 0.4 * 3 * y ** 2) * self.exp(x, y)
            - 0.02 * y * poly * self.exp(x, y)
            - 0.02 * (pol + y * poly - 0.02 * y * pol) * self.exp(x, y)
        )

    def eval(self, x, y):
        return (
            4 * x ** 2 - (y - 3 * x + 0.05 * x ** 3) ** 2 - 0.1 * y ** 4
        ) * self.exp(x, y)

    def exp(self, x, y):
        return np.exp(-0.01 * (x ** 2 + y ** 2))
