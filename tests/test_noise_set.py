from unittest import TestCase

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

from gp_explainer.gpx import Gpx
from gp_explainer.noise_set import NoiseSet


class TestNoiseSet(TestCase):

    def setUp(self) -> None:
        self.PRINT: bool = True
        self.NUN_SAMPLES: int = 250
        self.INSTANCE: int = 74
        x, y = make_moons(n_samples=500, noise=.1)
        self.x_train, \
            self.x_test, \
            self.y_train, \
            self.y_test = train_test_split(x, y, test_size=.3)

        self.clf = RandomForestClassifier(random_state=42)
        self.clf.fit(self.x_train, self.y_train)

        gpx = Gpx(self.clf.predict_proba,
                  x_train=self.x_train,
                  y_train=self.y_train,
                  feature_names=['x', 'y'],
                  num_samples=self.NUN_SAMPLES
                  )
        self.ns = NoiseSet(gpx)

    def test_create_noise_set(self):
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

        self.assertEqual(len(x_ns), self.NUN_SAMPLES + 2 * k)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES + 2 * k)

    def test_generate_data_around(self):
        x_ns, y_ns = self.ns.generate_data_around(self.x_test[self.INSTANCE, :])

        self.assertEqual(len(x_ns), self.NUN_SAMPLES)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES)

    def test_print_noise_k_neighbor(self):
        if self.PRINT:
            k = 4
            ns, \
                y_created, \
                k_neighbor, \
                k_distance, \
                each_distance, \
                each_class = self.ns.noise_k_neighbor(self.x_test[self.INSTANCE, :], k=k)

            x_train = np.append(self.x_train, ns, axis=0)
            y_train = np.append(self.y_train, self.clf.predict(ns))
            x_train = np.append(x_train, k_neighbor[0], axis=0)
            x_train = np.append(x_train, k_neighbor[1], axis=0)
            x_train = np.append(x_train, self.x_test[self.INSTANCE, :].reshape(1, -1), axis=0)

            a = x_train[:, 0]
            b = x_train[:, 1]
            c = y_train

            c = np.append(c, [2, 2, 2, 2, 3, 3, 3, 3, 4])
            c1 = np.ma.masked_where(c > 1, np.ones(len(c)) * 20)
            c2 = np.ma.masked_where(c < 2, np.ones(len(c)) * 200)
            c3 = np.ma.masked_where(c < 4, np.ones(len(c)) * 300)

            print(c1.shape, a.shape)

            plt.figure(figsize=(6, 4))
            plt.scatter(a, b, c=c, s=c1, cmap='viridis', label='data')
            plt.scatter(a, b, c=c, s=c2, marker='^', cmap='hot', label='k-neighbor')
            plt.scatter(a, b, c=c, s=c3, marker='X', cmap='hsv', label='instance')
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            pass
