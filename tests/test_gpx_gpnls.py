from unittest import TestCase

from pmlb import fetch_data
from pyoperon.sklearn import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from explainer.gpx import GPX


class TestNLS(TestCase):

    def setUp(self) -> None:

        self.NUN_SAMPLES: int = 250
        self.INSTANCE: int = 13

        x, y = fetch_data('210_cloud', return_X_y=True, local_cache_dir='./datasets')
        self.x_train, \
            self.x_test, \
            self.y_train, \
            self.y_test = train_test_split(x, y, test_size=.3)

        self.reg: RandomForestRegressor = RandomForestRegressor()
        self.reg.fit(self.x_train, self.y_train)
        self.my_operon = SymbolicRegressor(
            local_iterations=15,
            allowed_symbols='add,sub,mul,aq,exp,pow,constant,variable',
            generations=1000,
            n_threads=1,
            random_state=None,
            time_limit=60 * 60,  # 1 hours
            max_evaluations=1000,
            population_size=1000,
            max_length=25,
            tournament_size=3,
            epsilon=1e-5,
            reinserter='keep-best',
            offspring_generator='basic',
        )
        self.my_operon.fit(x, y)

        self.gpx = GPX(x=self.x_train,
                       y=self.y_train,
                       model_predict=self.reg.predict,
                       gp_model=self.my_operon,
                       noise_set_num_samples=self.NUN_SAMPLES)

        x, _, y, _ = self.gpx.instance_understanding(self.x_test[self.INSTANCE, :])

    def test_noise_set_generated(self):
            x_ns, y_ns = self.gpx.noise_set_generated(self.x_test[self.INSTANCE, :])

            self.assertEqual(len(x_ns), self.NUN_SAMPLES)
            self.assertEqual(len(y_ns), self.NUN_SAMPLES)

    def test_instance_understanding(self):
        x, _, y, _ = self.gpx.instance_understanding(self.x_test[self.INSTANCE, :])
        self.assertEqual(len(x), len(y))

    def test_get_string_expression(self):
        exp = self.gpx.get_string_expression()
        exp_op = self.my_operon.get_model_string(self.my_operon.model_)
        self.assertEqual(exp, exp_op)

    def test_show_tree(self):
        self.gpx.show_tree()
        exp = self.gpx.get_string_expression()
        exp_op = self.my_operon.get_model_string(self.my_operon.model_)
        self.assertEqual(exp, exp_op)

    def test_derivatives_generate(self):
        p = self.gpx.derivatives_generate(self.x_test[self.INSTANCE, :])
        self.gpx.show_tree()
        print(p)

    def test_image_base64(self):
        s = self.gpx.show_tree(is_base64=True)
        print(s)

    def test_numpy_derivatives_generate(self):
        p = self.gpx.derivatives_generate(self.x_test[self.INSTANCE, :], as_numpy=True)
        self.gpx.show_tree()
        print(p)

    def test_apply_differentials(self):
        p = self.gpx.derivatives_generate(self.x_test[self.INSTANCE, :], as_numpy=True)
        q = self.gpx.apply_differentials(self.x_test[self.INSTANCE, :])

        for k, v in p.items():
            assert q[k] == v




