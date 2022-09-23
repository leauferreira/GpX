from translate.adapter import Adapter


class GPAdapterFactory:

    def __init__(self, gp_obj):
        self.obj = gp_obj
        self.tool_name = str(type(gp_obj))

    def get_gp_obj(self):
        if self.tool_name.find("operon") >= 0:
            return Adapter(self.obj, expression_string=self.obj.get_model_string, my_name="operon")

        elif self.tool_name.find("gplearn") >= 0:
            return Adapter(self.obj, expression_string=lambda: self.obj._program, my_name="gplearn")

        elif self.tool_name.find("eckity") >= 0:
            # expression_string = self.obj.algorithm.population.get_best_individuals()
            return Adapter(self.obj, my_name="eckity")
        elif self.tool_name.find("translate.adapter.Adapter") >= 0:
            return self.obj
        else:
            raise ValueError(f"{self.tool_name} wasn't implemented")


