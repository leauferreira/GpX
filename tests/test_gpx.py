import unittest

from sklearn.neural_network import MLPClassifier

from gp_explainer.gpx import Gpx
from sklearn.datasets import make_moons

x_varied, y_varied = make_moons(n_samples=500, random_state=170)
model = MLPClassifier()
model.fit(x_varied, y_varied)
my_predict = model.predict


class TestGPX(unittest.TestCase):

    def test_gpx_classify(self):

        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, num_samples=250)
        gpx.explaining(x_varied[13, :])

        y_hat_gpx, _, _ = gpx.explaining(x_varied[13, :])
        y_hat_bb = my_predict(x_varied[13, :].reshape(1, -1))

        d = gpx.features_distribution()
        acc = gpx.understand()

        self.assertEqual(y_hat_gpx, y_hat_bb, "gpx fail in predict the black-box prediction")

        self.assertLess(d['X0'], d['X1'], 'Method features_distributions() output unexpected,  X1 greater than X0!')

        self.assertGreater(acc, 0.9, 'Accuracy decreasing in understand()  method of GPX class!!')


if __name__ == '__main__':
    unittest.main()
