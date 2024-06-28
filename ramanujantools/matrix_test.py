import sympy as sp
from sympy.abc import x, y

from ramanujantools import Matrix, simplify


def test_is_square():
    assert Matrix([[1, 2], [3, 4]]).is_square()
    assert not Matrix([1, 2, 3, 4]).is_square()


def test_gcd():
    a = 2 * 3 * 5
    b = 2 * 3 * 7
    c = 2 * 5 * 7
    d = 3 * 5 * 7
    m = Matrix([[a, b], [c, d]])
    m *= 11
    assert 11 == m.gcd()


def test_reduce():
    initial = Matrix([[2, 3], [5, 7]])
    gcd = sp.Rational(17, 13)
    m = gcd * initial
    assert m.gcd() == gcd
    assert m.reduce() == initial


def test_can_call_numerical_subs():
    m = Matrix([[x, 1], [y, 2]])
    assert not m._can_call_numerical_subs({x: 1})
    assert not m._can_call_numerical_subs({x: 1, y: y})
    assert m._can_call_numerical_subs({x: 17, y: 31})


def test_subs_degenerated():
    m = Matrix([[x, 1], [y, 2]])
    assert m == m.subs({x: x})
    assert m == m.subs({y: y})
    assert m == m.subs({x: x, y: y})


def test_subs_numerical():
    m = Matrix([[x, x**2], [13 + x, -x]])
    substitutions = {x: 5}
    assert m._can_call_numerical_subs(substitutions)
    assert Matrix([[5, 25], [18, -5]]) == m.subs(substitutions)


def test_subs_symbolic():
    m = Matrix([[x, x**2, 13 + x, -x]])
    expr = y**2 + x - 3
    assert Matrix([[expr, (expr) ** 2, 13 + expr, -expr]]) == m.subs({x: expr})


def test_as_polynomial():
    m = Matrix([[1, 1 / x], [0, 3 / (x**2 - x)]])
    polynomial_m = Matrix([[x * (x - 1), x - 1], [0, 3]])
    assert polynomial_m == m.as_polynomial()


def test_inverse():
    a = 5
    b = 2
    c = 3
    d = 7
    m = Matrix([[a, b], [c, d]])
    expected = Matrix([[d, -b], [-c, a]]) / (a * d - b * c)
    assert expected == m.inverse()


def test_singular_points_nonvariable():
    m = Matrix([[1, 2], [3, 4]])
    assert len(m.singular_points()) == 0


def test_singular_points_single_variable():
    m = Matrix([[1, 0], [1, (x - 1) * (x - 3)]])
    assert m.singular_points() == [{x: 1}, {x: 3}]


def test_singular_points_multi_variable():
    m = Matrix([[1, x], [1, y]])
    assert m.singular_points() == [{x: y}]


def test_walk_0():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 0, y: 1}, 0, {x: x, y: y}) == Matrix.eye(2)


def test_walk_1():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 1, y: 0}, 1, {x: x, y: y}) == m


def test_walk_list():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29, 53, 99]
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk(trajectory, iterations, start) == [
        m.walk(trajectory, i, start) for i in iterations
    ]


def test_walk_start_single_variable():
    iterations = [1, 2, 3, 4]
    m = Matrix([[0, x**2], [1, x + 1]])
    expected = m.walk({x: 1}, sum(iterations), {x: 1})
    actual = Matrix.eye(2)
    for i in range(len(iterations)):
        actual *= m.walk({x: 1}, iterations[i], {x: 1 + sum(iterations[0:i])})
    assert expected == actual


def test_walk_start_multi_variable():
    iterations = [1, 2, 3, 4]
    m = Matrix([[0, x**2], [1, y + 1]])
    starting_point = {x: 2, y: 3}
    trajectory = {x: 5, y: 7}
    expected = m.walk(trajectory, sum(iterations), starting_point)
    actual = Matrix.eye(2)
    for i in range(len(iterations)):
        actual *= m.walk(
            trajectory,
            iterations[i],
            {
                x: starting_point[x] + sum(iterations[0:i]) * trajectory[x],
                y: starting_point[y] + sum(iterations[0:i]) * trajectory[y],
            },
        )
    assert expected == actual


def test_walk_sequence():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29, 53, 99]
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk(trajectory, tuple(iterations), start) == m.walk(
        trajectory, set(iterations), start
    )


def test_walk_axis():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 1, y: 0}, 3, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 2, y: 1}) * m({x: 3, y: 1})
    )
    assert simplify(m.walk({x: 0, y: 1}, 5, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1})
        * m({x: 1, y: 2})
        * m({x: 1, y: 3})
        * m({x: 1, y: 4})
        * m({x: 1, y: 5})
    )


def test_walk_diagonal():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 1, y: 1}, 4, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 2, y: 2}) * m({x: 3, y: 3}) * m({x: 4, y: 4})
    )
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 4, y: 3}) * m({x: 7, y: 5})
    )


def test_walk_different_start():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 5, y: 7})) == simplify(
        m({x: 5, y: 7}) * m({x: 8, y: 9}) * m({x: 11, y: 11})
    )
