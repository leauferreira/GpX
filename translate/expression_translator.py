import sympy as sp


class Translator:

    def __init__(self,
                 gp_tool_name: str,
                 math_exp: str):
        self.gp_tool_name = gp_tool_name
        self.math_exp = math_exp

    def get_translation(self):
        if self.gp_tool_name.find("gplearn") >= 0:
            changes = {
                'sub': lambda x, y: x - y,
                'div': lambda x, y: x / y,
                'mul': lambda x, y: x * y,
                'add': lambda x, y: x + y,
                'neg': lambda x: -x,
                'pow': lambda x, y: x ** y,
                'cos': lambda x: sp.cos(x)
            }
            return sp.simplify(sp.sympify(str(self.math_exp), locals=changes))

        elif self.gp_tool_name.find("operon") >= 0:
            return sp.simplify(sp.sympify(str(self.math_exp)))

        elif self.gp_tool_name.find("eckity") >= 0:
            # TODO
            return "y - 3 + 5/x"

        else:
            raise ValueError(f"{self.tool_name} expression translator wasn't implemented")