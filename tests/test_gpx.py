import unittest

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from gp_explainer.gpx import Gpx
from sklearn.datasets import make_moons

x_varied, y_varied = make_moons(n_samples=500, random_state=170)
model = MLPClassifier()
model.fit(x_varied, y_varied)
my_predict = model.predict


class TestGPX(unittest.TestCase):

    def test_feature_sensitivity(self):

        x, y = make_moons(n_samples=1500, noise=.4, random_state=17)
        clf = MLPClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=17)
        clf.fit(x_train, y_train)

        gpx = Gpx(clf.predict, x_train=x, y_train=y, features_name=['x', 'y'])
        gpx.explaining(x_test[30, :])

        print(gpx.gp_model._program)

        gpx.feature_sensitivity()

    def test_gpx_classify(self):

        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, num_samples=250)

        y_hat_gpx, _, _ = gpx.explaining(x_varied[13, :])
        y_hat_bb = my_predict(x_varied[13, :].reshape(1, -1))

        d = gpx.features_distribution()

        acc = gpx.understand(metric='accuracy')

        print(gpx.gp_model._program)

        gpx.feature_sensitivity()

        self.assertEqual(y_hat_gpx, y_hat_bb, "gpx fail in predict the black-box prediction")

        self.assertLess(d['x_0'], d['x_1'], 'Method features_distributions() output unexpected,  X1 greater than X0!')

        self.assertGreater(acc, 0.9, 'Accuracy decreasing in understand()  method of GPX class!!')


if __name__ == '__main__':
    unittest.main()
