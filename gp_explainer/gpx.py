import numpy as np
import logging
from pathlib import Path
import pydotplus as pydotplus
from scipy.spatial import distance
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import accuracy_score, f1_score,  mean_squared_error, r2_score
import sympy as sp

from gp_explainer.noise_set import NoiseSet

fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
resource_path = 'unittest_gpx.log'
logging.basicConfig(filename=resource_path, level=logging.DEBUG,
                    filemode='w', format=fmt, datefmt='%d/%m/%Y %I:%M:%S %p')
logging.getLogger(__name__)

class Gpx:
    """
    Genetic Programming Explainer
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
                 feature_names=None,
                 random_state=None,
                 k_neighbor=None):
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
        self.x_train = x_train
        self.y_train = y_train
        self.x_train_measure = x_train_measure
        self.num_samples = num_samples
        self.problem = problem
        self.random_state = random_state
        self.gp_hyper_parameters = gp_hyper_parameters
        self.feature_names = feature_names
        self.x_around = None
        self.y_around = None

        if self.problem == 'classification':
            self.labels = np.unique(self.y_train)
        else:
            self.labels = None

        self.k_neighbor = k_neighbor
        self.gp_model = gp_model



    def program2sympy(self):
        program = str(self.gp_model._program)
        program = program.replace('add', 'Add')\
            .replace('mul', 'Mul')

        return program

    def gradient_analysis(self):

        sym = []
        df_partial = {}
        program = self.program2sympy()
        for name in self._feature_names:
            if name in program:
                sym.append(name)

        if sym:

            m_sym = ' '.join(sym)
            t_symbol = sp.symbols(m_sym)
            logging.info(self.gradient_analysis.__name__ + " - " + str(t_symbol))

            if isinstance(t_symbol, sp.Symbol):
                df = sp.diff(program, t_symbol)
                df_partial[t_symbol] = df
                logging.info(self.gradient_analysis.__name__ + ": t_symbol is sympy object ")

            else:

                for s in t_symbol:
                    df = sp.diff(program, s)
                    df_partial[s] = df
                    logging.info(self.gradient_analysis.__name__ + ":t_symbol is not sympy object")

            return df_partial

        else:
            logging.info("return 0")
            return 0

    def gp_fit(self):

        if self.x_around is not None and self.y_around is not None:

            if self.problem == 'regression':
                self.gp_model.fit(self.x_around, self.y_around)

            elif self.problem == 'classification' and len(self.labels) == 2:
                self.gp_model.fit(self.x_around, self.y_around[:, 0].reshape(-1))

            elif len(self.labels) > 2:
                threshold = 1/len(self.labels)
                for i, label in enumerate(self.labels):
                    y_aux = (self.y_around[:, i] > threshold) * 1
                    self.gp_model[label].fit(self.x_around, y_aux.ravel())

            else:
                raise ValueError("fit problem type unknown")

        else:
            raise ValueError("x_around and y_around must be created")

    def gp_prediction(self, x):

        if self.problem == 'regression':

            return self.gp_model.predict(x)

        elif self.problem == 'classification' and len(self.labels) == 2:

            y_proba = self.gp_model.predict(x)
            max_ = np.max(self.y_train)
            min_ = np.min(self.y_train)

            y = tuple(max_ if x <= .5 else min_ for x in y_proba)

            return np.array(y)

        elif self.problem == 'classification' and len(self.labels) > 2:

            m_y = np.zeros(shape=(x.shape[0], len(self.labels)))
            for i, label in enumerate(self.labels):
                y_aux = self.gp_model[label].predict(x)
                m_y[:, i] = y_aux.ravel()

            y_proba = np.argmax(m_y, axis=1)
            y = np.array([])
            for idx in y_proba:
                y = np.append(y, self.labels[idx])

            return y

        else:

            raise ValueError('predict problem type unknown')


    @property
    def gp_model(self):
        return self._gp_model

    @gp_model.setter
    def gp_model(self, gp_model):

        if self.problem == 'regression' or len(self.labels) == 2:

            if gp_model is None:
                f_names = self.gp_hyper_parameters.get('feature_names')
                if f_names is None:
                    self.gp_hyper_parameters['feature_names'] = self.feature_names
                self._gp_model = SymbolicRegressor(**self.gp_hyper_parameters)

            else:

                self._gp_model = gp_model
        else:

            dict_gp_model = {}

            for i in self.labels:

                if gp_model is None:
                    f_names = self.gp_hyper_parameters.get('feature_names')

                    if f_names is None:
                        self.gp_hyper_parameters['feature_names'] = self.feature_names

                    dict_gp_model[i] = SymbolicRegressor(**self.gp_hyper_parameters)

                else:

                    dict_gp_model[i] = gp_model

            self._gp_model = dict_gp_model

    @property
    def x_train_measure(self):
        return self._x_train_measure

    @x_train_measure.setter
    def x_train_measure(self, x_train_measure):
        if x_train_measure is None and self.x_train is not None:
            self._x_train_measure = np.std(self.x_train, axis=0) * .2
        else:
            self._x_train_measure = x_train_measure

    @property
    def gp_hyper_parameters(self):
        return self._gp_hyper_parameters

    @gp_hyper_parameters.setter
    def gp_hyper_parameters(self, gp_hyper_parameters):
        if gp_hyper_parameters is None:
            self._gp_hyper_parameters = {'population_size': 100,
                                         'generations': 100,
                                         'stopping_criteria': 0.00001,
                                         'p_crossover': 0.7,
                                         'p_subtree_mutation': 0.1,
                                         'p_hoist_mutation': 0.05,
                                         'p_point_mutation': 0.1,
                                         'const_range': (-1, 1),
                                         'parsimony_coefficient': 0.01,
                                         'init_depth': (2, 3),
                                         'n_jobs': -1,
                                         'random_state': self.random_state,
                                         'low_memory': True,
                                         'function_set': ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                                          'abs', 'neg', 'inv', 'max', 'min', 'sin',
                                                          'cos', 'tan')}
        else:
            self._gp_hyper_parameters = gp_hyper_parameters

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):

        if feature_names is None:
            self._feature_names = self.gp_hyper_parameters.get('feature_names')
        else:
            self._feature_names = feature_names

        if self._feature_names is None:
            self._feature_names = list('x_' + str(i) for i in range(self.x_train.shape[1]))

    def explaining(self, instance):
        ns = NoiseSet(self)
        self.x_around, self.y_around = ns.generate_data_around(instance)
        self.gp_fit()
        return self.gp_prediction(instance.reshape(1, -1))

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

        distribution = {}

        for name in self.feature_names:
            distribution[name] = 0

        for program in self.final_population:

            for name in self.feature_names:
                c = str(program).count(name)
                distribution[name] += c

        return distribution

    def proba_transform(self, y_proba):

        if len(self.labels) <= 2:

            max = np.max(self.y_train)
            min = np.min(self.y_train)
            y = tuple(max if x <= .5 else min for x in y_proba[:, 0])
            return np.array(y)

        else:

            y_aux = np.argmax(y_proba, axis=1)
            for i, j in enumerate(y_aux):
                y_aux[i] = self.labels[j]

            return y_aux

    def understand(self, instance=None, metric='report'):
        """

        :param instance:
        :param metric:
        :return:
        """

        d = {}
        if instance is None:
            y_hat_gpx = self.gp_prediction(self.x_around)
            y_hat_bb = self.y_around
        else:
            ns = NoiseSet(self)
            x_around, y_around = ns.noise_set(instance)
            y_hat_gpx = x_around
            y_hat_bb = self.proba_transform(self.predict(x_around))

        if self.problem == "classification":

            y_hat_bb = self.proba_transform(self.y_around)

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
                d['mse'] = mean_squared_error(y_hat_bb, y_hat_gpx)
                d['r2'] = r2_score(y_hat_bb, y_hat_gpx)
                d['rmse'] = mean_squared_error(y_hat_bb, y_hat_gpx, squared=False)
                d['rmspe'] = np.mean(np.sqrt(((y_hat_bb - y_hat_gpx)/y_hat_bb)**2))
                d['mape'] = np.mean(np.abs((y_hat_bb - y_hat_gpx)/y_hat_bb))
                return d

            elif metric == 'r2':
                return r2_score(y_hat_bb, y_hat_gpx)

            elif metric == 'mse':
                return mean_squared_error(y_hat_bb, y_hat_gpx)

            elif metric == 'rmse':
                return mean_squared_error(y_hat_bb, y_hat_gpx, squared=False)

            elif metric == 'rmspe':
                return np.mean(np.sqrt(((y_hat_bb - y_hat_gpx)/y_hat_bb)**2))

            elif metric == 'mape':
                return np.mean(np.abs((y_hat_bb - y_hat_gpx)/y_hat_bb))

            else:
                logging.error('understand can not be used with {}'.format(metric))
                raise ValueError('understand can not be used with {}'.format(metric))

        else:
            logging.error('understand can not be used with problem type as {}'.format(metric))
            raise ValueError('understand can not be used with problem type as {}'.format(metric))

    def max_min_matrix(self, noise_set=None, dist_type='upward', noise_range=100):

        """
        Provide a matrix for explainer analysis. Must be for a class of utilities.

        :param noise_set:
        :param dist_type:
        :param noise_range:
        :return:
        """

        if noise_set is None:

            v_min = np.min(self.x_around, axis=0)
            v_max = np.max(self.x_around, axis=0)
            rows, columns = self.x_around.shape
            mmm = np.zeros(shape=(noise_range, columns))

        else:

            v_min = np.min(noise_set, axis=0)
            v_max = np.max(noise_set, axis=0)
            rows, columns = noise_set.shape
            mmm = np.zeros(shape=(noise_range, columns))

        if dist_type == 'upward':
            f_dist = np.linspace
        elif dist_type == 'uniform':
            f_dist = np.random.uniform

        for i, (min_, max_) in enumerate(zip(v_min, v_max)):
            actual = f_dist(min_, max_, noise_range)
            mmm[:, i] = actual.ravel()

        return mmm

    def feature_sensitivity(self, verbose=False):

        mmm = self.max_min_matrix()
        sz = self.x_around.shape[0]
        n_samples = sz // 10
        idx = np.random.randint(sz, size=n_samples)

        samples = self.x_around[idx, :]

        header: bool = False

        feature_dict = {}

        for p in self.gp_model._program.program:

            if isinstance(p, int):

                is_sensitive: bool = False

                if not header and verbose:
                    pg_str = str(self.gp_model._program)
                    print("|{:^41}|".format(pg_str))
                    print("|{:^10}|{:^10}|{:^20}|".format("FEATURE", "NOISE", "SENSITIVITY"))
                    header = True

                aux = samples.copy()

                np_sens = np.zeros(shape=100)

                for i, rate in enumerate(mmm[:, p]):

                    aux[:, p] = rate
                    y_true = self.gp_prediction(samples)
                    y_eval = self.gp_prediction(aux)

                    sens = 1 - np.mean((y_true == y_eval) * 1)

                    if sens > 0 and verbose:
                        print("|{:^10.8}|{:^10.2f}|{:^20.2f}|".format(self.feature_names[p], rate, sens))

                    if sens > 0:
                        np_sens[i] = sens
                        is_sensitive = True

                if is_sensitive:
                    feature_dict[self.feature_names[p]] = (mmm[:, p], np_sens)

        return feature_dict