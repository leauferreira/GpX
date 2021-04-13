import numpy as np
import logging


class Utils:

    @staticmethod
    def max_min_matrix(noise_set, dist_type='upward', noise_range=100):

        """
        Provide a matrix for explainer analysis. Must be for a class of utilities.

        :param noise_set: set created around instance numpy nd_array
        :param dist_type: upward and uniform
        :param noise_range: default 100 samples
        :return: numpy array
        """

        v_min = np.min(noise_set, axis=0)
        v_max = np.max(noise_set, axis=0)
        rows, columns = noise_set.shape
        mmm = np.zeros(shape=(noise_range, columns))

        if dist_type == 'upward':
            f_dist = np.linspace
        elif dist_type == 'uniform':
            f_dist = np.random.uniform
        else:
            logging.debug(Utils.max_min_matrix.__name__ + f' - dist_type {dist_type} is not implemented')
            raise ValueError(f'dist_type {dist_type} is not implemented')

        for i, (min_, max_) in enumerate(zip(v_min, v_max)):
            actual = f_dist(min_, max_, noise_range)
            mmm[:, i] = actual.ravel()

        return mmm

    @staticmethod
    def translate_to_sympy(program):
        pass

