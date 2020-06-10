from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
import pydotplus as pydotplus
import numpy as np


class Gpx:
    """
    Genetic Programming Explainable main class. This class will provide interpretability by a single
    prediction made by a black-box model.
    """

    def __init__(self,
                 predict,
                 x_train_measure=None,
                 x_train=None,
                 y_train=None,
                 num_samples=1000,
                 problem='classification',
                 gp_model=None,
                 gp_hyper_parameters=None,
                 features_name=None):
        """

        :param predict: prediction function from black-box model. y_hat = predict(instance)
        :param x_train_measure: std for each feature in x_train, for instance scale = np.std(x_train)
        :param x_train: input training set
        :param y_train: target training set
        :param num_samples: size of noise set
        :param problem: type of problem (default = classification)
        :param gp_model: Genetic programming model (default provided by gplearn)
        :param gp_hyper_parameters: dictionary with hyper parameters of GP model
        :param feature_names: list with all features Names
        """
        self.predict = predict
        self.x_train_measure = x_train_measure
        self.x_train = x_train
        self.y_train = y_train
        self.num_samples = num_samples
        self.problem = problem
        self.feature_names = features_name

        if gp_model is None:

            if gp_hyper_parameters is None:

                self.gp_hyper_parameters = {'population_size': 300,
                                            'generations': 100,
                                            'stopping_criteria': 0.00001,
                                            'p_crossover': 0.7,
                                            'p_subtree_mutation': 0.1,
                                            'p_hoist_mutation': 0.05,
                                            'p_point_mutation': 0.1,
                                            'const_range': (-100, 100),
                                            'parsimony_coefficient': 0.01,
                                            'init_depth': (2, 3),
                                            'feature_names': self.feature_names}

            else:

                self.gp_hyper_parameters = gp_hyper_parameters

            if problem == 'classification':

                self.gp_model = SymbolicClassifier(**self.gp_hyper_parameters)

            else:

                self.gp_model = SymbolicRegressor(**self.gp_hyper_parameters)

        if x_train_measure is None and x_train is not None:
            self.x_train_measure = np.std(x_train, axis=0) * .1

    def create_noise_set(self, instance):

        d = len(instance)
        x_created = np.random.normal(instance, scale=self.x_train_measure, size=(self.num_samples, d))
        y_created = self.predict(x_created)

        if self.problem == 'regression':

            return x_created, y_created

        else:

            y_min = np.min(y_created)
            y_max = np.max(y_created)

            if y_max != y_min:

                return x_created, y_created

            else:

                i_want = np.where(self.y_train != y_max)[0]
                x_other_class = self.x_train[i_want, :]
                cut = np.floor(self.num_samples * .2)
                cut = int(cut)
                x_created = np.concatenate((x_created, x_other_class[:cut, :]), axis=0)
                y_created = self.predict(x_created)

                return x_created, y_created

    def explaining(self, instance):

        x_around, y_around = self.create_noise_set(instance)
        self.gp_model.fit(x_around, y_around)
        y_hat = self.gp_model.predict(instance.reshape(1, -1))

        return y_hat, x_around, y_around

    def make_graphviz_model(self):

        return pydotplus.graphviz.graph_from_dot_data(self.gp_model._program.export_graphviz())







