import unittest
import numpy as np
from gp_explainable.gpx import Gpx
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier


class TestGPX(unittest.TestCase):

    def test_gpx_init(self):
        clf = MLPRegressor(hidden_layer_sizes=(100, 100, 100))
        gpx = Gpx(clf, problem='regression')
        self.assertEqual(type(gpx.gp_hyper_parameters), type(dict()))

    def test_gpx_classify(self):
        n_samples = 1500
        random_state = 170
        x_varied, y_varied = make_blobs(n_samples=n_samples,
                                        cluster_std=[1.0, 2.5, 0.5],
                                        random_state=random_state)
        model = RandomForestClassifier(n_estimators=25)
        model.fit(x_varied, y_varied)
        my_predict = model.predict
        scale = np.std(x_varied, axis=0) * .35
        gpx = Gpx(my_predict, x_train_measure=scale, x_train=x_varied, y_train=y_varied)
        gpx.explaining(x_varied[13, :])

        y_hat_gpx, _, _ = gpx.explaining(x_varied[13, :])
        y_hat_bb = my_predict(x_varied[13, :].reshape(1, -1))
        self.assertEqual(y_hat_gpx, y_hat_bb)


if __name__ == '__main__':
    unittest.main()
