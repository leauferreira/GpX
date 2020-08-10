import unittest

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from gp_explainer.gpx import Gpx
from sklearn.datasets import make_moons


class TestGPX(unittest.TestCase):

    def test_understand(self):
        x, y = make_moons(n_samples=1500, noise=.4, random_state=17)
        clf = MLPClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=17)
        clf.fit(x_train, y_train)

        gpx = Gpx(clf.predict, x_train=x, y_train=y, features_name=['x', 'y'])
        gpx.explaining(x_test[30, :])

        try:
            u = gpx.understand(metric='loss')
        except ValueError as e:
            gpx.logger.exception(e)
            u = gpx.understand(metric='accuracy')
            self.assertGreater(u, .9)


    def test_feature_sensitivity(self):

        x, y = make_moons(n_samples=1500, noise=.4, random_state=17)
        clf = MLPClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=17)
        clf.fit(x_train, y_train)

        gpx = Gpx(clf.predict, x_train=x, y_train=y, features_name=['x', 'y'])
        gpx.explaining(x_test[30, :])

        dict_sens = gpx.feature_sensitivity()

        for key in dict_sens:
            sens = dict_sens[key][0]
            gpx.logger.info('soma sensibilidade: {}'.format(np.sum(sens)))
            self.assertGreater(np.sum(sens), 1)

    def test_gpx_classify(self):

        x_varied, y_varied = make_moons(n_samples=500, random_state=170)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict

        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, num_samples=250)

        y_hat_gpx, _, _ = gpx.explaining(x_varied[13, :])
        y_hat_bb = my_predict(x_varied[13, :].reshape(1, -1))

        d = gpx.features_distribution()

        acc = gpx.understand(metric='accuracy')


        self.assertEqual(y_hat_gpx, y_hat_bb, "gpx fail in predict the black-box prediction")

        gpx.logger.info('distribution: x_0: {} / x_1: {}'.format(d['x_0'], d['x_1']))
        self.assertLess(d['x_0'], d['x_1'], 'Method features_distributions() output unexpected,  X1 greater than X0!')

        gpx.logger.info('test accuracy: {}'.format(acc))
        self.assertGreater(acc, 0.9, 'Accuracy decreasing in understand()  method of GPX class!!')


if __name__ == '__main__':
    unittest.main()
