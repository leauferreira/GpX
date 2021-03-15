from unittest import TestCase

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from gp_explainer.gpx import Gpx
from gp_explainer.noise_set import NoiseSet


class TestNoiseSet(TestCase):

    def setUp(self) -> None:
        self.NUN_SAMPLES: int = 100
        self.INSTANCE: int = 17
        x, y = make_moons(n_samples=500, noise=.1)
        self.x_train, \
            self.x_test, \
            self.y_train, \
            self.y_test = train_test_split(x, y, test_size=.3)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(self.x_train, self.y_train)

        gpx = Gpx(clf.predict_proba,
                  x_train=self.x_train,
                  y_train=self.y_train,
                  feature_names=['x', 'y'],
                  num_samples=self.NUN_SAMPLES
                  )
        self.ns = NoiseSet(gpx)

    def test_noise_set(self):
        x_ns, y_ns = self.ns.create_noise_set(self.x_test[self.INSTANCE, :])

        self.assertEqual(len(x_ns), self.NUN_SAMPLES)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES)

    def test_noise_set(self):

        x_ns, y_ns = self.ns.noise_set(self.x_test[self.INSTANCE, :])

        self.assertEqual(len(x_ns), self.NUN_SAMPLES)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES)

    def test_noise_k_neighbor(self):
        k = 4
        x_ns, y_ns, _, _, _, _ = self.ns.noise_k_neighbor(self.x_test[self.INSTANCE, :], k=k)

        self.assertEqual(len(x_ns), self.NUN_SAMPLES + 2*k)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES + 2*k)

    def test_generate_data_around(self):
        x_ns, y_ns = self.ns.generate_data_around(self.x_test[self.INSTANCE, :])

        self.assertEqual(len(x_ns), self.NUN_SAMPLES)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES)
