from unittest import TestCase

from translate.expression_translator import Translator


class TestTranslator(TestCase):
    def test_get_translation(self):

        operon_str = "y - 3 + 5/x"
        gplearn_str = "sub(div(5, x), sub(3, y))"

        t_operon = Translator("operon", operon_str).get_translation()
        t_gplearn = Translator("gplearn", gplearn_str).get_translation()

        self.assertEqual(t_gplearn, t_operon)


