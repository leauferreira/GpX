from unittest import TestCase

import numpy as np

from gplearn.genetic import SymbolicRegressor
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

from explainer.gpx_classification import GPXClassification


class TestGPXClassification(TestCase):

    def test_predict(self):

        x_varied, y_varied = make_moons(n_samples=1000, random_state=170)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict

        gp_hyper_parameters = {'population_size': 200,
                               'generations': 200,
                               'stopping_criteria': 0.0000001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.01,
                               # 'init_depth': (2, 3),
                               'n_jobs': -1,
                               'low_memory': True,
                               'function_set': ('add', 'sub', 'mul', 'div')}

        my_gplearn = SymbolicRegressor(**gp_hyper_parameters)
        my_gplearn.fit(x_varied, y_varied)
        gpx = GPXClassification(model_predict=my_predict, x=x_varied, gp_model=my_gplearn)
        gpx.instance_understanding(x_varied[3, :])
        x, y = make_moons(15)
        y_hat = gpx.predict(x)

        print(y)
        print(y_hat)
        print(np.mean((y == y_hat)*1))
        self.assertGreaterEqual(np.mean((y == y_hat)*1), 0.5)

