import sympy as sp
import os
from sympy.abc import x, y
from os.path import dirname
from os.path import join as path_join

from ramanujan import Matrix
from ramanujan.cmf import CMF, ffbar

c0 = sp.Symbol("c0")
c1 = sp.Symbol("c1")
c2 = sp.Symbol("c2")
c3 = sp.Symbol("c3")

CACHED_CMFS_PATH = path_join(dirname(__file__), 'cached_cmfs')


def e():
    return CMF(
        Matrix([[1, -y - 1], [-1, x + y + 2]]), Matrix([[0, -y - 1], [-1, x + y + 1]])
    )


def pi():
    return CMF(
        Matrix([[x, -x], [-y, 2 * x + y + 1]]),
        Matrix([[1 + y, -x], [-1 - y, x + 2 * y + 1]]),
    )


def zeta3():
    return CMF(
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
        potential_cache_file=path_join(CACHED_CMFS_PATH, 'zeta3_cmf_first_50x50_cells.pkl')
    )


def cmf1():
    return ffbar.construct(c0 + c1 * (x + y), c2 + c3 * (x - y))


def cmf2():
    return ffbar.construct(
        (2 * c1 + c2) * (c1 + c2)
        - c3 * c0
        - c3 * ((2 * c1 + c2) * (x + y) + (c1 + c2) * (2 * x + y))
        + c3**2 * (2 * x**2 + 2 * x * y + y**2),
        c3 * (c0 + c2 * x + c1 * y) - c3**2 * (2 * x**2 - 2 * x * y + y**2),
    )


def cmf3_1():
    return ffbar.construct(
        -((c0 + c1 * (x + y)) * (c0 * (x + 2 * y) + c1 * (x**2 + x * y + y**2))),
        (c0 + c1 * (-x + y)) * (c0 * (x - 2 * y) - c1 * (x**2 - x * y + y**2)),
    )


def cmf3_2():
    return ffbar.construct(
        -(x + y) * (c0**2 + 2 * c1**2 * (x**2 + x * y + y**2)),
        (x - y) * (c0**2 + 2 * c1**2 * (x**2 - x * y + y**2)),
    )


def cmf3_3():
    return ffbar.construct(
        (x + y)
        * (c0**2 - c0 * c1 * (x - y) - 2 * c1**2 * (x**2 + x * y + y**2)),
        (c0 + c1 * (x - y)) * (3 * c0 * (x - y) + 2 * c1 * (x**2 - x * y + y**2)),
    )
