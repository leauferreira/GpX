
from neighborhood.noise_set import NoiseSet

import numpy as np


class GPX:

    def __init__(self,
                 x,
                 y,
                 model_predict,
                 gp_model,
                 noise_set_num_samples=100
                 ):
        self.x = x
        self.y = y
        self.model_predict = model_predict
        self.gp_model = gp_model
        self.noise_set_num_samples = noise_set_num_samples

    def noise_set_generated(self, instance):
        info_data = np.std(self.x, axis=0) * .2
        ns = NoiseSet(self.model_predict, info_data, self.noise_set_num_samples)
        return ns.noise_set(instance)

    def instance_understanding(self, instance):
        x_around, y_around = self.noise_set_generated(instance)
        self.gp_model.fit(x_around, y_around)

    def get_string_expression(self):
        return self.gp_model.get_model_string(2)
