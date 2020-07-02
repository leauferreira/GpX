import unittest

from sklearn.neural_network import MLPClassifier

from gp_explainable.gpx import Gpx
from sklearn.datasets import make_moons


class TestGPX(unittest.TestCase):

    def test_gpx_classify(self):
        n_samples = 500
        random_state = 170
        x_varied, y_varied = make_moons(n_samples=n_samples,
                                        random_state=random_state)

        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict
        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, num_samples=250)
        gpx.explaining(x_varied[13, :])

        y_hat_gpx, _, _ = gpx.explaining(x_varied[13, :])
        y_hat_bb = my_predict(x_varied[13, :].reshape(1, -1))

        d = gpx.features_distribution()

        self.assertEqual(y_hat_gpx, y_hat_bb)
        self.assertLess(d['X0'], d['X1'], 'features distributions of X1 greater than X0!!!')


if __name__ == '__main__':
    unittest.main()
