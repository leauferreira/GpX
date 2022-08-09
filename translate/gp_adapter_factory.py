from translate.adapter import Adapter


class GPAdapterFactory:

    def __init__(self, gp_obj):
        self.obj = gp_obj
        self.name = str(type(gp_obj))

    def get_gp_obj(self):
        if self.name.find("operon") >= 0:
            return Adapter(self.obj, expression_string=self.obj.get_model_string)

        elif self.name.find("gplearn") >= 0:
               return Adapter(self.obj, expression_string=lambda: self.obj._program)

        else:
            raise ValueError(f"{self.name} wasn't implemented")


