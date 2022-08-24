import graphviz


class TreeExplanation:

    def __init__(self, str_math_exp):
        self.graph_source = graphviz.Source(str_math_exp)

    def generate_image(self, directory: str = None, filename: str = None, view: bool = True):
        self.graph_source.render(view=view, filename=filename, directory=directory)

