import unittest

import numpy as np
import sympy as sp

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import load_boston, make_moons, load_breast_cancer, load_iris, load_wine
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler

from gp_explainer.gpx import Gpx

import logging
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# from gp_explainer.noise_set import NoiseSet

# fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# logging.basicConfig(filename="test_gpx.log", level=logging.INFO,
#                     filemode='w', format=fmt, datefmt='%d/%m/%Y %I:%M:%S %p')
# logging.getLogger(__name__)


class TestGPX(unittest.TestCase):

    def test_gradient(self):

        INSTANCE: int = 15
        wine = load_wine()
        X, y = load_wine(return_X_y=True)

        X = X[y != 2]
        y = y[y != 2]

        clf = RandomForestClassifier()
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=.5, test_size=.5, random_state=42)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        clf.fit(x_train, y_train)

        gp_hyper_parameters = {'population_size': 100,
                               'generations': 100,
                               'stopping_criteria': 0.00001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.0001,
                               'init_depth': (3, 6),
                               'n_jobs': -1,
                               'function_set': ('add', 'mul', 'sin', 'cos', 'tan', 'sqrt'),
                               'random_state': 42,
                               }

        gpx = Gpx(clf.predict_proba,
                  x_train=x_train,
                  gp_hyper_parameters=gp_hyper_parameters,
                  y_train=y_train,
                  feature_names=wine.feature_names,
                  num_samples=5000,
                  k_neighbor=10)

        gpx.explaining(scaler.transform(x_test[INSTANCE, :].reshape(1, -1)))

        prog = gpx.program2sympy()

        print(sp.latex(sp.sympify(prog)))

        print(prog)

        part_dic = gpx.gradient_analysis()

        print('\n\n')
        print(part_dic)

        my_subs = zip(wine.feature_names, x_test[INSTANCE, :])
        my_subs_list = []

        for s_name, s_value in my_subs:
            if s_name in prog:
                my_subs_list.append((s_name, s_value))

        print('\n\n')

        y_0 = []
        y_1 = []
        names = []
        for k, v in part_dic.items():
            partial = v.subs(my_subs_list)
            print('feature: {}  gradient: {}'.format(k, partial))
            if partial >= 0:
                y_0.append(partial)
            else:
                y_1.append(partial)
            names.append(str(k).upper())
        x_0 = range(len(y_0))
        x_1 = range(len(y_0), len(part_dic))

        fig, ax = plt.subplots()
        ax.barh(x_0, y_0, color='b')
        ax.barh(x_1, y_1, color='r')
        width = 0.3
        ind = np.arange(len(part_dic))
        ax.set_yticks(ind + width / 2)
        ax.set_yticklabels(names, minor=False)
        # for i, v in enumerate(y_0 + y_1):
        #     ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
        plt.show()

    def test_multi_class_wine(self):

        INSTANCE: int = 17
        iris = load_wine()
        X, y = load_wine(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

        scaler = Normalizer()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        gp_hyper_parameters = {'population_size': 50,
                               'generations': 50,
                               'stopping_criteria': 0.0001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.0005,
                               'init_depth': (3, 6),
                               'n_jobs': -1,
                               'function_set': ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                                'abs', 'neg', 'inv', 'max', 'min', 'sin',
                                                'cos', 'tan')
                               }

        gpx = Gpx(clf.predict_proba,
                  gp_hyper_parameters=gp_hyper_parameters,
                  x_train=X_train,
                  y_train=y_train,
                  feature_names=iris.feature_names,
                  num_samples=1000,
                  k_neighbor=5)

        y_hat = gpx.explaining(scaler.transform(X_test[INSTANCE, :].reshape(-1, 1)))

        x_around = gpx.x_around

        gpx_y = gpx.gp_prediction(x_around)
        bb_y = clf.predict(x_around)

        gpx.logger.info('Multiclass: gpx_understand {}'.format(gpx.understand(metric='f1')))
        gpx.logger.info('Multiclass gpx_y:{} / bb_y {}'.format(gpx_y, bb_y))
        gpx.logger.info('test_understand mult-class accuracy {}'.format(accuracy_score(gpx_y, bb_y)))

    def test_multi_class(self):

        INSTANCE: int = 20
        iris = load_iris()
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

        scaler = Normalizer()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        gp_hyper_parameters = {'population_size': 100,
                               'generations': 100,
                               'stopping_criteria': 0.0001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.0005,
                               'init_depth': (3, 6),
                               'n_jobs': -1,
                               'random_state': 42,
                               'function_set': ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                                'abs', 'neg', 'inv', 'max', 'min', 'sin',
                                                'cos', 'tan')
                               }

        gpx = Gpx(clf.predict_proba,
                  gp_hyper_parameters=gp_hyper_parameters,
                  x_train=X_train,
                  y_train=y_train,
                  feature_names=iris.feature_names,
                  num_samples=500,
                  k_neighbor=50,
                  random_state=42)

        y_hat = gpx.explaining(scaler.transform(X_test[INSTANCE, :].reshape(-1, 1)))

        x_around = gpx.x_around

        gpx_y = gpx.gp_prediction(x_around)
        bb_y = clf.predict(x_around)

        gpx.logger.info('Multiclass gpx_y:{} / bb_y {}'.format(gpx_y, bb_y))
        gpx.logger.info('test_understand mult-class accuracy {}'.format(accuracy_score(gpx_y, bb_y)))



    def test_feature_names(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        f_names = load_breast_cancer().feature_names

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        gp_hyper_parameters = {'population_size': 200,
                               'generations': 200,
                               'stopping_criteria': 0.0001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.0005,
                               'init_depth': (3, 6),
                               'n_jobs': -1,
                               'function_set': ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                                'abs', 'neg', 'inv', 'max', 'min', 'sin',
                                                'cos', 'tan'),
                               'feature_names': f_names}

        gpx = Gpx(model.predict_proba, x_train=X_train, y_train=y_train, random_state=42,
                  gp_hyper_parameters=gp_hyper_parameters)
        gpx.explaining(X_test[30, :])

        print(gpx.gp_model._program)

    def test_grafic_sensibility(self):
        INSTANCE: int = 74
        x, y = make_moons(n_samples=1500, noise=.4, random_state=17)
        clf = MLPClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=17)
        clf.fit(x_train, y_train)

        gpx = Gpx(clf.predict_proba, x_train=x, y_train=y, random_state=42, feature_names=['x', 'y'])
        gpx.explaining(x_test[INSTANCE, :])

        x, y = gpx.x_around[:, 0], gpx.x_around[:, 1]
        y_proba = gpx.proba_transform(gpx.y_around)

        resolution = 0.02
        x1_min, x1_max = x.min() - 1, x.max() + 1
        x2_min, x2_max = y.min() - 1, y.max() + 1
        xm1, xm2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z_bb = gpx.gp_prediction(np.array([xm1.ravel(), xm2.ravel()]).T)

        fig, ax = plt.subplots()
        ax.set_xlim(x1_min, x1_max)
        ax.set_xlim(x2_min, x2_max)
        scat = plt.scatter(x, y, y_proba)

        def func(data):
            k, j = data
            scat.set_offsets(k)
            scat.set_array(j)

        mmm = gpx.max_min_matrix(noise_range=10)

        gen = []
        for n in mmm[:, 0]:
            aux = gpx.x_around.copy()
            aux[:, 0] = n
            gen.append((aux.copy(), gpx.gp_prediction(aux.copy())))

        animation = ani.FuncAnimation(fig, func, gen, interval=200, save_count=200)

        plt.contourf(xm1, xm2, Z_bb.reshape(xm1.shape), alpha=0.4)
        plt.scatter(x, y, c=y_proba)

        plt.show()

        writergif = ani.PillowWriter(fps=5)
        animation.save('sens_x_2.gif',  writer=writergif)

        sens_gpx = gpx.feature_sensitivity()
        print(sens_gpx)

    def test_gpx_regression(self):

        INSTANCE: int = 13
        reg = RandomForestRegressor()
        x, y = load_boston(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=42)
        reg.fit(x_train, y_train)

        gpx = Gpx(predict=reg.predict, x_train=x_train, y_train=y_train, problem='regression', random_state=42)
        gpx.explaining(x_test[INSTANCE, :])
        y_hat = reg.predict(x_test)
        mse = mean_squared_error(y_test, y_hat)

        d = gpx.features_distribution()

        # self.assertEqual(max(list(d.values())), d['x_2'])

        self.assertLess(gpx.understand(metric='mse'), mse,
                        '{} mse greater than understand (local mse)'.format(self.test_gpx_regression.__name__))

    def test_understand(self):
        x, y = make_moons(n_samples=1500, noise=.4, random_state=17)
        clf = MLPClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=17)
        clf.fit(x_train, y_train)

        gpx = Gpx(clf.predict_proba, x_train=x, y_train=y, feature_names=['x', 'y'])
        gpx.explaining(x_test[30, :])

        y = clf.predict_proba(x_test)
        gpx.logger.info( gpx.proba_transform(y))

        try:
            u = gpx.understand(metric='loss')
        except ValueError as e:
            gpx.logger.exception(e)
            u = gpx.understand(metric='accuracy')
            gpx.logger.info('test_understand accuracy {}'.format(u))
            self.assertGreater(u, .9, 'test_understand accuracy {}'.format(u))

    def test_feature_sensitivity(self):
        x, y = make_moons(n_samples=1500, noise=.4, random_state=17)
        clf = MLPClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=17)
        clf.fit(x_train, y_train)

        gpx = Gpx(clf.predict_proba, x_train=x, y_train=y, random_state=42, feature_names=['x', 'y'])
        gpx.explaining(x_test[30, :])

        dict_sens = gpx.feature_sensitivity()

        for key in dict_sens:
            sens = dict_sens[key][0]
            gpx.logger.info('test_feature_sensitivity soma sensibilidade: {}'.format(np.sum(sens)))
            self.assertGreater(np.sum(sens), 1)

    def test_features_distribution(self):
        x_varied, y_varied = make_moons(n_samples=500, random_state=170)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict_proba

        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, random_state=42, num_samples=250)

        gpx.explaining(x_varied[13, :])
        d = gpx.features_distribution()

        gpx.logger.info('distribution-> program:{} / x_0: {} / x_1: {}'.format(gpx.gp_model._program,
                                                                               d['x_0'], d['x_1']))
        self.assertLess(d['x_0'], d['x_1'], 'Method features_distributions() output unexpected, x_0 greater than x_1!')

    def test_gpx_classify(self):

        MY_INST = 33
        x_varied, y_varied = make_moons(n_samples=500, random_state=170)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict_proba

        gpx = Gpx(my_predict, x_train=x_varied, y_train=y_varied, random_state=42, num_samples=250)

        y_hat_gpx = gpx.explaining(x_varied[MY_INST, :])
        y_hat_bb = my_predict(x_varied[MY_INST, :].reshape(1, -1))

        acc = gpx.understand(metric='accuracy')

        gpx.logger.info('{} / y_hat_gpx: {} / y_hat_bb: {}'. format(self.test_gpx_classify.__name__,
                                                                    type(y_hat_gpx), type(y_hat_bb)))

        self.assertEqual(np.sum(y_hat_gpx), np.sum(y_hat_bb), "gpx fail in predict the black-box prediction")

        gpx.logger.info('test accuracy: {}'.format(acc))
        self.assertGreater(acc, 0.9, 'Accuracy decreasing in understand()  method of GPX class!!')


if __name__ == '__main__':
    unittest.main()
