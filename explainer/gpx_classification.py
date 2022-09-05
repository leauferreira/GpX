import graphviz
import numpy as np
import sympy as sp

from explain.show_explanation import TreeExplanation
from neighborhood.noise_set import NoiseSet
from translate.expression_translator import Translator
from translate.gp_adapter_factory import GPAdapterFactory


class GPXClassification:

    def __init__(self,
                 x,
                 model_predict,
                 gp_model,
                 noise_set_num_samples=100
                 ):
        self.x = x
        self.model_predict = model_predict
        self.gp_model = GPAdapterFactory(gp_model).get_gp_obj()
        self.noise_set_num_samples = noise_set_num_samples

    def noise_set_generated(self, instance):
        info_data = np.std(self.x, axis=0) * 1
        ns = NoiseSet(self.model_predict, info_data, self.noise_set_num_samples)
        return ns.noise_set(instance)

    def instance_understanding(self, instance):
        """

        @param instance:
        @return:
        """
        x_around, y_around = self.noise_set_generated(instance)
        self.gp_model.fit(x_around, y_around[:, 1] * 100)

        return x_around, y_around

    def get_string_expression(self) -> str:
        return self.gp_model.expression_string()

    def show_tree(self,
                  directory: str = None,
                  filename: str = None,
                  view: bool = True,
                  cleanup: bool = False,
                  ) -> graphviz.Source:

        """

        @param directory:
        @param filename:
        @param view:
        @param cleanup:
        @return:
        """
        sp_exp = Translator(gp_tool_name=self.gp_model.my_name, math_exp=self.get_string_expression()).get_translation()
        dot_sp = sp.dotprint(sp.N(sp_exp, 3))
        te = TreeExplanation(dot_sp)
        te.generate_image(view=view, filename=filename, directory=directory, cleanup=cleanup)

        return te.graph_source

    def predict(self, x):
        y_hat = self.gp_model.predict(x) / 100

        print('yhat: ', y_hat)
        return (1 / (1+np.exp(-y_hat)) >= 0.5)*1
