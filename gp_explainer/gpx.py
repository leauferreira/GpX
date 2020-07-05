from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
import pydotplus as pydotplus
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_squared_log_error, mean_squared_error


class Gpx:
    """
    Genetic Programming Explainable main class. This class will provide interpretability by a single
    prediction made by a black-box model.
    """

    def __init__(self,
                 predict,
                 x_train,
                 y_train,
                 x_train_measure=None,
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
        self.final_population = None
        self.predict = predict
        self.x_train_measure = x_train_measure
        self.x_train = x_train
        self.y_train = y_train
        self.num_samples = num_samples
        self.problem = problem
        self.features_names = features_name
        self._x_around = None
        self._y_around = None

        if gp_model is None:

            if gp_hyper_parameters is None:

                self.gp_hyper_parameters = {'population_size': 100,
                                            'generations': 100,
                                            'stopping_criteria': 0.00001,
                                            'p_crossover': 0.7,
                                            'p_subtree_mutation': 0.1,
                                            'p_hoist_mutation': 0.05,
                                            'p_point_mutation': 0.1,
                                            'const_range': (-1, 1),
                                            'parsimony_coefficient': 0.01,
                                            'init_depth': (2, 3),
                                            'random_state': 42,
                                            'n_jobs': -1,
                                            'low_memory': True,
                                            'function_set': ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
                                                             'inv', 'max', 'min', 'sin', 'cos', 'tan'),
                                            'feature_names': self.features_names}

            else:

                self.gp_hyper_parameters = gp_hyper_parameters

            if problem == 'classification':

                self.gp_model = SymbolicClassifier(**self.gp_hyper_parameters)

            else:

                self.gp_model = SymbolicRegressor(**self.gp_hyper_parameters)

        if x_train_measure is None and x_train is not None:
            self.x_train_measure = np.std(x_train, axis=0) * .2

    def create_noise_set(self, instance):
        """
        Create a noise set around a instance that will be explain

        :param instance: numpy array with size equals to number of features in problem.
        :return: x, y created around instance (y is predict by a black box model)
        """

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

    def noise_set(self, instance):
        """
        Create a noise set around a instance that will be explain

        :param instance: numpy array with size equals to number of features in problem.
        :return: x, y created around instance (y is predict by a black box model predict)
        """

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

                for i, yt in enumerate(self.y_train):
                    if yt != y_max:
                        x_created[i, :] = self.x_train[i, :]
                        y_created[i] = self.y_train[i]
                        break

                return x_created, y_created

    def explaining(self, instance):

        self._x_around, self._y_around = self.noise_set(instance)
        self.gp_model.fit(self._x_around, self._y_around)
        y_hat = self.gp_model.predict(instance.reshape(1, -1))

        return y_hat, self._x_around, self._y_around

    def make_graphviz_model(self):
        """
        Make a graphviz model of final tree structure

        :return: graphviz model
        """

        return pydotplus.graphviz.graph_from_dot_data(self.gp_model._program.export_graphviz())

    def features_distribution(self):
        """
        Count all  feature occurrence in the last population.

        :return: dictionary with key = feature and value = total occurrence of that feature in gp final population.
        """

        self.final_population = self.gp_model._programs[-1]

        if self.features_names is None:
            names = ['X' + str(i) for i in range(self.x_train.shape[1])]
        else:
            names = self.features_names

        distribution = {}

        for name in names:
            distribution[name] = 0

        for program in self.final_population:

            for name in names:
                c = str(program).count(name)
                distribution[name] += c

        return distribution

    def understand(self, instance=None, metric='report'):

        d = {}
        if instance is None:
            y_hat_gpx = self.gp_model.predict(self._x_around)
            y_hat_bb = self._y_around
        else:
            x_around, y_around = self.noise_set(instance)
            y_hat_gpx = self.gp_model.predict(x_around)
            y_hat_bb = self.predict(x_around)

        if self.problem == "classification":
            d['accuracy'] = accuracy_score(y_hat_bb, y_hat_gpx)
            d['f1'] = f1_score(y_hat_bb, y_hat_gpx, average='micro')

            if metric == 'report':
                d['accuracy'] = accuracy_score(y_hat_bb, y_hat_gpx)
                d['f1'] = f1_score(y_hat_bb, y_hat_gpx, average='micro')
                return d

            elif metric == 'accuracy':
                return accuracy_score(y_hat_bb, y_hat_gpx)

            elif metric == 'f1':
                return f1_score(y_hat_bb, y_hat_gpx, average='micro')

            else:
                raise ValueError('understand can not be used with {}'.format(metric))

        elif self.problem == 'regression':

            if metric == 'report':
                d['msle'] = mean_squared_log_error(y_hat_bb, y_hat_gpx)
                d['mse'] = mean_squared_error(y_hat_bb, y_hat_gpx)
                return d

            elif metric == "msle":
                return mean_squared_log_error(y_hat_bb, y_hat_gpx)

            elif metric == 'mse':
                return mean_squared_error(y_hat_bb, y_hat_gpx)

            else:
                raise ValueError('understand can not be used with {}'.format(metric))

        else:
            raise ValueError('understand can not be used with problem type as {}'.format(metric))

