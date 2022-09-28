from unittest import TestCase

import sympy as sp
from sympy.vector import gradient
from explain.show_explanation import TreeExplanation
from explain.show_explanation import ExtractGradient


class TestTreeExplanation(TestCase):

    def test_generate_image(self):
        str_math_test = '((-1.28) + (0.738 * ((1.77 * X4) - ((((-5) * X3) / (sqrt(1 + (1.77 * X4) ^ 2))) ' \
                        '/ (sqrt(1 + ((-2) * X4) ^ 2))))))'

        sp_exp = sp.sympify(str_math_test)
        dot_sp = sp.dotprint(sp.N(sp_exp, 3))
        te = TreeExplanation(dot_sp)
        te.generate_image()


class TestExtractGradient(TestCase):
    def test_get_symbols(self):
        str_math_test = '((-1.28) + (0.738 * ((1.77 * X4) - ((((-5) * X3) / (sqrt(1 + (1.77 * X4) ^ 2))) ' \
                        '/ (sqrt(1 + ((-2) * X4) ^ 2))))))'

        sp_exp = sp.sympify(str_math_test)

        eg = ExtractGradient(sp_exp)

        (a, b) = eg.get_symbols()

        self.assertEqual({str(a), str(b)}, {'X4', 'X3'})

    def test_do_the_derivatives(self):
        str_math_test = 'x*y + x**2'

        sp_exp = sp.sympify(str_math_test)

        eg = ExtractGradient(sp_exp)

        print(eg.do_the_derivatives())

        print(gradient(sp_exp))

    def test_partial_derivatives(self):

        str_math_test = 'x*y + x**2 + sin(z) - z*y'
        sp_exp = sp.sympify(str_math_test)

        eg = ExtractGradient(sp_exp)

        p = sp_exp.subs([('x', 1), ('y', 1), ('z', 1)]).evalf()
        print(p)

        print(eg.do_the_derivatives())

        r = eg.partial_derivatives({'x': 1, 'y': 1, 'z': 1})
        print(r)






