import graphviz
from sklearn.model_selection import train_test_split

from explain.show_explanation import TreeExplanation, ExtractGradient
from translate.expression_translator import Translator
from translate.gp_adapter_factory import GPAdapterFactory
from neighborhood.noise_set import NoiseSet

import numpy as np
import sympy as sp


class GPX:

    def __init__(self,
                 x,
                 y,
                 model_predict,
                 gp_model,
                 noise_set_num_samples=100,
                 info_data_rate=1,
                 feature_names=None):

        self.x = x
        self.y = y
        self.model_predict = model_predict
        self.gp_model = gp_model
        self.noise_set_num_samples = noise_set_num_samples
        self.info_data_rate = info_data_rate
        self.feature_names = feature_names

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        if feature_names is None:
            self._feature_names = np.array([f'X{i+1}' for i in range(self.x.shape[1])])
        else:
            self._feature_names = feature_names


    def noise_set_generated(self, instance):
        info_data = np.std(self.x, axis=0) * self.info_data_rate
        ns = NoiseSet(self.model_predict, info_data, self.noise_set_num_samples)
        return ns.noise_set(instance)

    def instance_understanding(self, instance):
        """

        @param instance:
        @return:
        """
        x_around, y_around = self.noise_set_generated(instance)
        x_train, x_test, y_train, y_test = train_test_split(x_around,
                                                            y_around,
                                                            train_size=0.7,
                                                            test_size=0.3,
                                                            shuffle=True)
        self.gp_model.fit(x_train, y_train)
        self.gp_model = GPAdapterFactory(self.gp_model).get_gp_obj()

        return x_train, x_test, y_train, y_test

    def get_string_expression(self) -> str:
        if self.gp_model.my_name == "operon":
            return self.gp_model.expression_string(self.gp_model.obj.model_)
        else:
            return self.gp_model.expression_string()

    def show_tree(self,
                  directory: str = None,
                  filename: str = None,
                  view: bool = True,
                  cleanup: bool = False,
                  is_base64: bool = False
                  ):

        sp_exp = Translator(gp_tool_name=self.gp_model.my_name, math_exp=self.get_string_expression()).get_translation()
        dot_sp = sp.dotprint(sp.N(sp_exp, 3))
        te = TreeExplanation(dot_sp)
        if not is_base64:
            te.generate_image(view=view, filename=filename, directory=directory, cleanup=cleanup)
            return te.graph_source
        else:
            return te.generate_base64_image()

    def derivatives_generate(self, instance, as_numpy=False):
        sp_exp = Translator(gp_tool_name=self.gp_model.my_name, math_exp=self.get_string_expression()).get_translation()
        eg = ExtractGradient(sp_exp, self.feature_names)
        if as_numpy:
            return eg.partial_derivatives(instance, as_numpy)
        else:
            inst_dict = {'X' + str(i+1): value for i, value in enumerate(instance)}
            return eg.partial_derivatives(inst_dict)
