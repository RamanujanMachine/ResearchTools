from typing import List
import sympy

from ramanujan.cmf.high_d_cmf import CMF
from ramanujan.matrix import Matrix
from ramanujan.simplify_object import simplify
from ramanujan.generic_polynomial import GenericPolynomial

x, y, z, w, q = sympy.symbols('x y z w q')


def high_dim_CMF_example(
        name: str, variables: List[sympy.Symbol], t: sympy.Symbol,
        symmetric_matrix: Matrix,
        pcf_mx: Matrix, pcf_mt: Matrix) -> CMF:
    cmf = CMF({
        v: symmetric_matrix({t: v}) for v in variables
    })

    x = variables[0]
    mx = cmf.matrices[x]
    cc_str = ','.join(str(v) for v in variables)
    print(f'M_X({cc_str}) = {mx}')

    u = Matrix([[1, mx[0, 0]], [0, mx[1, 0]]])
    print(f'U={u}')
    # Taking co boundary means moving to Ms -> u^-1 * Ms * u(s->s+1), where s in variables.
    pcf_version = cmf.coboundary(u)

    print(f'check assertions {name}:')
    y = variables[1]
    assert simplify(pcf_mx - pcf_version.matrices[x]) == Matrix([[0, 0], [0, 0]])
    assert simplify(pcf_mt({t: y}) - pcf_version.matrices[y]) == Matrix([[0, 0], [0, 0]])
    print(f'{name} example validated')

    return cmf


def high_dim_CMF_example_4D():
    # symmetric version:
    cc = [x, y, z, w]
    symm = GenericPolynomial.symmetric_polynomials(*cc)
    t = sympy.Symbol('t')
    symmetric_matrix = Matrix([
        [t * t + symm[2], -1],
        [symm[4], t * t]
    ])

    # pcf version
    pcf_mx = Matrix([
        [0, -(x + 1) * (x + y) * (x + z) * (x + w)],
        [1, (y * z + z * w + w * y) + (y + z + w + x) * (x + 1) + (x + 1) ** 2]
    ])
    # Note that the bottom right can also be written as
    # symm2 + symm1 + x**2 + (x+1)**2

    # compute the t-direction where t in y,z,w.
    pcf_mt = Matrix([
        [t ** 2 - x ** 2, -(x + 1) * (x + y) * (x + z) * (x + w)],
        [1, symm[2] + symm[1] + x ** 2 + t ** 2]
    ])
    # for a choice t=y , can also write it as
    pcf_my = Matrix([
        [y ** 2 - x ** 2, -(x + 1) * (x + y) * (x + z) * (x + w)],
        [1, (x * z + z * w + w * x) + (x + y + z + w) * (y + 1) + x ** 2]
    ])

    # construction and validations
    cmf = high_dim_CMF_example(
        name='4D', variables=cc, t=t,
        symmetric_matrix=symmetric_matrix,
        pcf_mx=pcf_mx, pcf_mt=pcf_mt)

    return cmf


# high_dim_CMF_example_4D()


def high_dim_CMF_example_5D():
    # symmetric version:
    cc = [x, y, z, w, q]
    symm = GenericPolynomial.symmetric_polynomials(*cc)
    t = sympy.Symbol('t')
    symmetric_matrix = Matrix([
        [t ** 2 + symm[2], (t ** 2 + symm[2]) * (t - symm[1]) - symm[4]],
        [1, t ** 2 + (t - symm[1])]])
    # = Matrix([
    #         [t ** 2 + symm[2], -symm[4]],
    #         [1,                t ** 2  ]
    # ]) * Matrix([
    #         [1, t-symm[1] ],
    #         [0, 1         ]
    # ])
    #
    # So it has the determinant
    # t**4 + symm[2]*t**2 + symm[4]

    # pcf version
    pcf_mx = Matrix([
        [0, -(x + q) * (x + y) * (x + z) * (x + w)],
        [1, symm[2] + x ** 2 + (x + 1) ** 2]
    ])
    # note that you can also write:
    #    (x + q) * (x + y) * (x + z) * (x + w) = x**4 + symm[2]*x**2 + symm[4]
    pcf_mt = Matrix([
        [t ** 2 - x ** 2, -(x + y) * (x + z) * (x + w) * (x + q)],
        [1, symm[2] + t ** 2 + x ** 2]
    ])

    # construction and validations
    cmf = high_dim_CMF_example(
        name='5D', variables=cc, t=t,
        symmetric_matrix=symmetric_matrix,
        pcf_mx=pcf_mx, pcf_mt=pcf_mt)
    return cmf


# high_dim_CMF_example_5D()


def high_dim_CMF_example_6D():
    cc = list(sympy.symbols('c:5'))
    x = cc[0]
    symm = GenericPolynomial.symmetric_polynomials(*cc)
    symm_s = GenericPolynomial.symmetric_polynomials(*cc[1:5])
    t = sympy.Symbol('t')

    # the following are matrices in x=c[0], c[1], c[2], c[3], c[4], y
    b = -(x + cc[1]) * (x + cc[2]) * (x + cc[3]) * (x + cc[4])
    a = x**2 + (x + 1) ** 2 + symm[2] + (y + symm_s[1] - 1) * y

    # note that in general, there functions are not polynomial in y, c1, c2, c3, c4, but are polynomial in x
    f = x * (y + symm[1]) + (- symm_s[3] + (y + symm_s[1]) * (y**2 + y*symm_s[1] + symm_s[2])) / (symm_s[1] + 2 * y)
    bf = -x ** 2 + x * y - (y ** 3 + symm_s[1] * y ** 2 + y * symm_s[2] + symm_s[3]) / (symm_s[1] + 2 * y)
    # when all the roots are zero, then these are the standard functions of the zeta 2 cmf, namely
    # f = x**2 + x*y + y**2 / 2
    # bf = -x ** 2 + x * y - y ** 2 / 2

    F = x**2 + t*(t+y) + symm[2] + y*symm[1] + y**2
    BF = -x**2 + x*y + t*(t+y)

    mx = Matrix([
        [0, b],
        [1, a]
    ])

    my = Matrix([
        [bf, b],
        [1, f]
    ])

    # mt is a "symmetric" matrix in a sense that we can replace t with one of the generic roots c1, c2, c3, c4
    mt = Matrix([
        [BF, b],
        [1, F]
    ])

    # check that the x=c1, c2, c3, c4, c5 directions form a 5D conservative matrix field for every y
    directions = {v: mt({t: v}) for v in cc[1:]}
    directions[x] = mx
    roots_cmf = CMF(directions)
    print('5D conservativeness done')

    # check that for any fixed roots, the x,y directions form a conservative matrix field
    xy_cmf = CMF({x: mx, y: my})
    print('x,y direction conservativeness done')


high_dim_CMF_example_6D()
