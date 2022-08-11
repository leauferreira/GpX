from unittest import TestCase

from gplearn.genetic import SymbolicRegressor
from pmlb import fetch_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from explainer.gpx import GPX


class TestGPX(TestCase):

    def setUp(self) -> None:
        self.PRINT: bool = True
        self.NUN_SAMPLES: int = 250
        self.INSTANCE: int = 13
        x, y = fetch_data('210_cloud', return_X_y=True, local_cache_dir='./datasets')
        self.x_train, \
            self.x_test, \
            self.y_train, \
            self.y_test = train_test_split(x, y, test_size=.3)

        self.reg: RandomForestRegressor = RandomForestRegressor()
        self.reg.fit(self.x_train, self.y_train)

        gp_hyper_parameters = {'population_size': 100,
                               'generations': 100,
                               'stopping_criteria': 0.00001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.01,
                               'init_depth': (2, 3),
                               'n_jobs': -1,
                               'low_memory': True,
                               'function_set': ('add', 'sub', 'mul', 'div')}

        self.my_gplearn = SymbolicRegressor(**gp_hyper_parameters)
        self.my_gplearn.fit(X=self.x_train, y=self.y_train)

        self.gpx = GPX(x=self.x_train, y=self.y_train, model_predict=self.reg.predict, gp_model=self.my_gplearn,
                       noise_set_num_samples=self.NUN_SAMPLES)

    def test_noise_set_generated(self):
        x_ns, y_ns = self.gpx.noise_set_generated(self.x_test[self.INSTANCE, :])

        self.assertEqual(len(x_ns), self.NUN_SAMPLES)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES)

    def test_instance_understanding(self):
        self.gpx.instance_understanding(self.x_test[self.INSTANCE, :])

    def test_get_string_expression(self):
        exp = self.gpx.get_string_expression()
        print(exp)



