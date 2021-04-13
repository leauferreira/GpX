from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from gp_explainer.gpx import Gpx
from gp_explainer.noise_set import NoiseSet

import time

NUN_SAMPLES: int = 250
INSTANCE: int = 74
x, y = make_moons(n_samples=500, noise=.1)
x_train, \
    x_test, \
    y_train, \
    y_test = train_test_split(x, y, test_size=.3)

clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)

gpx = Gpx(clf.predict_proba,
          x_train=x_train,
          y_train=y_train,
          feature_names=['x', 'y'],
          num_samples=NUN_SAMPLES
          )


ns = NoiseSet(gpx)


def test_k_neighbor_opt(benchmark):
    benchmark.pedantic(ns.k_neighbor_adapter, args=(x_test[13, :], 4), iterations=3, rounds=30)


def test_k_neighbor_nor(benchmark):
    benchmark.pedantic(ns.noise_k_neighbor, args=(x_test[13, :], 4), iterations=3, rounds=30)


def test_k_neighbor_raw(benchmark):
    benchmark.pedantic(NoiseSet.k_neighbor, args=(x_test[13, :], 4, x_train,
                                                  y_train,
                                                  gpx.labels,
                                                  gpx.num_samples),
                       iterations=3,
                       rounds=30)
