import unittest
import py_data_manager.data_manager as dm


class TestDataManager(unittest.TestCase):

    def test_all_classifications(self):
        data = dm.get_all_classifications()

        self.assertEqual(len(data), 10)
        x1, y1 = data[dm.Classification.monks_problems_2.name].get_np_x_y()
        x2, y2 = dm.data_classification_factory(dm.Classification.monks_problems_2).get_np_x_y()
        self.assertEqual(y1.all(), y2.all())

    def test_all_regression(self):
        data = dm.get_all_regressions()
        self.assertEqual(len(data), 10)


if __name__ == '__main__':
    unittest.main()
