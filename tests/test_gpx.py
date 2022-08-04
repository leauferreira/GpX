from unittest import TestCase

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

        self.reg: RandomForestRegressor = RandomForestRegressor(random_state=42)
        self.reg.fit(self.x_train, self.y_train)

        self.gpx = GPX(x=self.x_train, y=self.y_train, model_predict=self.reg.predict, gp_model=0,
                       noise_set_num_samples=self.NUN_SAMPLES)

    def test_noise_set_generated(self):
        x_ns, y_ns = self.gpx.noise_set_generated(self.x_test[self.INSTANCE, :])

        self.assertEqual(len(x_ns), self.NUN_SAMPLES)
        self.assertEqual(len(y_ns), self.NUN_SAMPLES)
