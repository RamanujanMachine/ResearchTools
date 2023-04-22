import sympy as sp
from sympy.abc import x, y

from matrix import Matrix
from cmf import CMF
from ffbar import ffbar

c0 = sp.Symbol("c0")
c1 = sp.Symbol("c1")
c2 = sp.Symbol("c2")
c3 = sp.Symbol("c3")

e = CMF(Matrix([[1, -y - 1], [-1, x + y + 2]]), Matrix([[0, -y - 1], [-1, x + y + 1]]))

pi = CMF(
    Matrix([[x, -x], [-y, 2 * x + y + 1]]),
    Matrix([[1 + y, -x], [-1 - y, x + 2 * y + 1]]),
)

zeta3 = CMF(
    Matrix(
        [
            [0, -(x**3)],
            [(x + 1) ** 3, x**3 + (x + 1) ** 3 + 2 * y * (y - 1) * (2 * x + 1)],
        ]
    ),
    Matrix(
        [
            [-(x**3) + 2 * x**2 * y - 2 * x * y**2 + y**3, -(x**3)],
            [x**3, x**3 + 2 * x**2 * y + 2 * x * y**2 + y**3],
        ]
    ),
)

cmf1 = ffbar(c0 + c1 * (x + y), c2 + c3 * (x - y))
cmf2 = ffbar(
    (2 * c1 + c2) * (c1 + c2)
    - c3 * c0
    - c3 * ((2 * c1 + c2) * (x + y) + (c1 + c2) * (2 * x + y))
    + c3**2 * (2 * x**2 + 2 * x * y + y**2),
    c3 * (c0 + c2 * x + c1 * y) - c3**2 * (2 * x**2 - 2 * x * y + y**2),
)
