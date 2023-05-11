import graphviz

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
                 noise_set_num_samples=100
                 ):

        self.x = x
        self.y = y
        self.model_predict = model_predict
        self.gp_model = gp_model
        self.noise_set_num_samples = noise_set_num_samples

    def noise_set_generated(self, instance):
        info_data = np.std(self.x, axis=0) * .25
        ns = NoiseSet(self.model_predict, info_data, self.noise_set_num_samples)
        return ns.noise_set(instance)

    def instance_understanding(self, instance):
        """

        @param instance:
        @return:
        """
        x_around, y_around = self.noise_set_generated(instance)
        self.gp_model.fit(x_around, y_around)
        self.gp_model = GPAdapterFactory(self.gp_model).get_gp_obj()

        return x_around, y_around

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




    def derivatives_generate(self, instance):
        sp_exp = Translator(gp_tool_name=self.gp_model.my_name, math_exp=self.get_string_expression()).get_translation()
        eg = ExtractGradient(sp_exp)
        inst_dict = {'X' + str(i+1): value for i, value in enumerate(instance)}
        return eg.partial_derivatives(inst_dict)
