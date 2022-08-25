from unittest import TestCase
import os
import sympy as sp
from explain.show_explanation import TreeExplanation


class TestTreeExplanation(TestCase):

    def test_generate_image(self):
        str_math_test = '((-1.28) + (0.738 * ((1.77 * X4) - ((((-5) * X3) / (sqrt(1 + (1.77 * X4) ^ 2))) ' \
                        '/ (sqrt(1 + ((-2) * X4) ^ 2))))))'

        sp_exp = sp.sympify(str_math_test)
        dot_sp = sp.dotprint(sp.N(sp_exp, 3))
        te = TreeExplanation(dot_sp)
        te.generate_image()



