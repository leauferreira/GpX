from unittest import TestCase
import pydm.pydm as pdm



class TestDataManager(TestCase):

    def test_get_all_multiclass(self):
        data = pdm.get_all_multiclass()
        len_mc = len(pdm.MultiClass)
        len_data = len(data)

        self.assertEqual(len_mc, len_data)
