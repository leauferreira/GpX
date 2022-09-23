from unittest import TestCase

import numpy as np

from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.sklearn_compatible.sk_classifier import SKClassifier
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_encodings.gp.tree.functions import f_add, f_mul, f_sub, f_div, f_neg, f_sqrt, f_log, f_abs, f_inv, f_max, \
    f_min
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_avg_worst_size_tree_statistics import BestAverageWorstSizeTreeStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker

# Adding your own functions
from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator

from explainer.gpx import GPX
from explainer.gpx_classification import GPXClassification


class TestGPXClassification(TestCase):

    def test_predict(self):

        x_varied, y_varied = make_moons(n_samples=1000)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict_proba

        gp_hyper_parameters = {'population_size': 200,
                               'generations': 200,
                               'stopping_criteria': 0.0000001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.01,
                               # 'init_depth': (2, 3),
                               'n_jobs': -1,
                               'low_memory': True,
                               'function_set': ('add', 'sub', 'mul', 'div')}

        my_gplearn = SymbolicRegressor(**gp_hyper_parameters)
        my_gplearn.fit(x_varied, y_varied)
        gpx = GPXClassification(model_predict=my_predict, x=x_varied, gp_model=my_gplearn, noise_set_num_samples=300)
        gpx.instance_understanding(x_varied[3, :])
        x, y = make_moons(50)
        y_hat = gpx.predict(x)

        print(y)
        print(y_hat)
        print(np.mean((y == y_hat)))

        gpx.show_tree()

        self.assertGreaterEqual(np.mean((y == y_hat)*1), 0.5)

    def test_classifier_eckity(self):
        X, y = make_moons(n_samples=1000)
        model = MLPClassifier()
        model.fit(X, y)
        my_predict = model.predict

        terminal_set = create_terminal_set(X)

        # Define function set
        function_set = [f_add, f_mul, f_sub, f_div]  # f_sqrt, f_log, f_abs, f_neg, f_inv, f_max, f_min

        # Initialize SimpleEvolution instance
        algo = SimpleEvolution(
            Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, 4),
                                                            terminal_set=terminal_set,
                                                            function_set=function_set,
                                                            bloat_weight=0.0001),
                          population_size=1000,
                          evaluator=ClassificationEvaluator(),
                          # maximization problem (fitness is accuracy), so higher fitness is better
                          higher_is_better=True,
                          elitism_rate=0.05,
                          # genetic operators sequence to be applied in each generation
                          operators_sequence=[
                              SubtreeCrossover(probability=0.9, arity=2),
                              SubtreeMutation(probability=0.2, arity=1)
                          ],
                          selection_methods=[
                              # (selection method, selection probability) tuple
                              (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                          ]
                          ),
            breeder=SimpleBreeder(),
            max_workers=1,
            max_generation=1000,
            # optimal fitness is 1, evolution ("training") process will be finished when best fitness <= threshold
            termination_checker=ThresholdFromTargetTerminationChecker(optimal=1, threshold=0.03),
            statistics=BestAverageWorstSizeTreeStatistics()
        )
        # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
        classifier = SKClassifier(algo)

        # classifier.fit(X, y)

        gpx = GPX(model_predict=my_predict, x=X, y=y, gp_model=classifier, noise_set_num_samples=300)
        gpx.instance_understanding(X[3, :])

        x, y = make_moons(50)
        y_hat = gpx.gp_model.predict(x)

        print(y)
        print(y_hat)
        print(np.mean((y == y_hat)))

        self.assertGreaterEqual(np.mean((y == y_hat)*1), 0.6)

    def test_classifier_gplearn(self):

        x_varied, y_varied = make_moons(n_samples=1000)
        model = MLPClassifier()
        model.fit(x_varied, y_varied)
        my_predict = model.predict_proba

        gp_hyper_parameters = {'population_size': 200,
                               'generations': 200,
                               'stopping_criteria': 0.0000001,
                               'p_crossover': 0.7,
                               'p_subtree_mutation': 0.1,
                               'p_hoist_mutation': 0.05,
                               'p_point_mutation': 0.1,
                               'const_range': (-1, 1),
                               'parsimony_coefficient': 0.01,
                               # 'init_depth': (2, 3),
                               'n_jobs': -1,
                               'low_memory': True,
                               'function_set': ('add', 'sub', 'mul', 'div')}

        my_gplearn_cls = SymbolicClassifier(**gp_hyper_parameters)



