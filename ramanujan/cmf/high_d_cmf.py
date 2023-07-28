from typing import Dict
import sympy
from sympy.core.symbol import Symbol

from ramanujan.matrix import Matrix
from ramanujan.simplify_object import simplify


class CMF:

    def _verify(self, s1: Symbol, s2: Symbol):
        m1 = self.matrices[s1]
        m2 = self.matrices[s2]
        m = m1 * m2({s1: s1 + 1}) - m2 * m1({s2: s2 + 1})
        if simplify(m) != Matrix([[0, 0], [0, 0]]):
            raise ValueError(f'The matrices at directions ({s1}, {s2}) are not conservative')

    def __init__(self, matrices: Dict[Symbol, Matrix]):
        self.matrices = matrices

        # verify conservativeness
        symbols = list(matrices.keys())
        n = len(symbols)
        for i in range(n):
            for j in range(i):
                self._verify(symbols[i], symbols[j])

    def coboundary(self, co_matrix: Matrix):
        det = co_matrix[0,0]*co_matrix[1,1] - co_matrix[0,1]*co_matrix[1,0]
        co_matrix_inv = Matrix([
            [co_matrix[1,1], -co_matrix[0,1]],
            [-co_matrix[1,0], co_matrix[0,0]]
        ])
        co_matrix_inv /= det

        new_matrices = {
            symbol: co_matrix_inv*matrix*co_matrix({symbol: symbol+1}) for symbol, matrix in self.matrices.items()}
        return CMF(new_matrices)

