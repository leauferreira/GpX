from unittest import TestCase
import pydm.pydm as pdm



class TestDataManager(TestCase):

    def test_get_all_multiclass(self):
        data = pdm.get_all_multiclass()
        len_mc = len(pdm.MultiClass)
        len_data = len(data)
        self.assertEqual(len_mc, len_data)

    def test_get_all_classification(self):
        data = pdm.get_all_classifications()
        len_cls = len(pdm.Classification)
        len_data = len(data)
        self.assertEqual(len_cls, len_data)

    def test_get_all_regression(self):
        data = pdm.get_all_regressions()
        len_reg = len(pdm.Regression)
        len_data = len(data)
        self.assertEqual(len_reg, len_data)


