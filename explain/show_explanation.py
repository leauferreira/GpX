import base64

import graphviz
import sympy as sp


class TreeExplanation:

    def __init__(self, str_math_exp):
        self.graph_source = graphviz.Source(str_math_exp)

    def generate_image(self,
                       directory: str = None,
                       filename: str = None,
                       view: bool = True,
                       cleanup: bool = False,):
        """

        @param directory:
        @param filename:
        @param view:
        @param cleanup:
        """
        self.graph_source.render(view=view, filename=filename, directory=directory, cleanup=cleanup)

    def generate_base64_image(self):
        graph_data = self.graph_source.pipe(format='png')
        return base64.b64encode(graph_data).decode('utf-8')


class ExtractGradient:

    def __init__(self, str_math_exp):
        self.math_exp = str_math_exp

    def get_symbols(self):
        return self.math_exp.free_symbols

    def do_the_derivatives(self):
        return {str(s): sp.diff(self.math_exp, s) for s in self.get_symbols()}

    def partial_derivatives(self, instance):

        partial = self.do_the_derivatives()
        results = {}
        for s, f in partial.items():
            sym_f = f.free_symbols
            replacements = [(str(a), instance[str(a)]) for a in sym_f]
            results[s] = f.subs(replacements).evalf()

        return results



