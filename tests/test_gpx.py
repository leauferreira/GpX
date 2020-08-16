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

        gpx = Gpx(clf.predict_proba, x_train=x, y_train=y, feature_names=['x', 'y'])
        gpx.explaining(x_test[30, :])

        y = clf.predict_proba(x_test)
        gpx.logger.info( gpx.proba_transform(y))

        try:
            u = gpx.understand(metric='loss')
        except ValueError as e:
            gpx.logger.exception(e)
            u = gpx.understand(metric='accuracy')
            gpx.logger.info('test_understand accuracy {}'.format(u))
            self.assertGreater(u, .9, 'test_understand accuracy {}'.format(u))

    def test_feature_sensitivity(self):

        x, y = make_moons(n_samples=1500, noise=.4, random_state=17)
        clf = MLPClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=17)
        clf.fit(x_train, y_train)

        gpx = Gpx(clf.predict_proba, x_train=x, y_train=y, random_state=42, feature_names=['x', 'y'])
        gpx.explaining(x_test[30, :])

        dict_sens = gpx.feature_sensitivity()

        for key in dict_sens:
            sens = dict_sens[key][0]
            gpx.logger.info('test_feature_sensitivity soma sensibilidade: {}'.format(np.sum(sens)))
            self.assertGreater(np.sum(sens), 1)

    def test_features_distribution(self):
        x_varied, y_varied = make_moons(n_samples=500, random_state=170)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict_proba

        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, random_state=42, num_samples=250)

        gpx.explaining(x_varied[13, :])
        d = gpx.features_distribution()

        gpx.logger.info('distribution-> program:{} / x_0: {} / x_1: {}'.format(gpx.gp_model._program,
                                                                               d['x_0'], d['x_1']))
        self.assertLess(d['x_0'], d['x_1'], 'Method features_distributions() output unexpected, x_0 greater than x_1!')

    def test_gpx_classify(self):

        MY_INST = 33
        x_varied, y_varied = make_moons(n_samples=500, random_state=170)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict_proba

        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, random_state=42, num_samples=250)

        y_hat_gpx = gpx.explaining(x_varied[MY_INST, :])
        y_hat_bb = my_predict(x_varied[MY_INST, :].reshape(1, -1))

        acc = gpx.understand(metric='accuracy')

        gpx.logger.info('{} / y_hat_gpx: {} / y_hat_bb: {}'. format(self.test_gpx_classify.__name__,
                                                                    type(y_hat_gpx), type(y_hat_bb)))

        self.assertEqual(np.sum(y_hat_gpx), np.sum(y_hat_bb), "gpx fail in predict the black-box prediction")

        gpx.logger.info('test accuracy: {}'.format(acc))
        self.assertGreater(acc, 0.9, 'Accuracy decreasing in understand()  method of GPX class!!')


if __name__ == '__main__':
    unittest.main()
