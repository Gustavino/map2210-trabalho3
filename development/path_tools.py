import os
import scipy.io

import numpy as np


def sorted_matrices_and_paths(path):
    matrices = []
    matrices_paths = []

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            matrices_paths.append(os.path.join(dirname, filename))
            matrices.append(filename)

    dense_matrices = [np.copy(scipy.io.mmread(matrix).todense()) for matrix in matrices_paths]
    matrices_dimensions = [len(matrix) for matrix in dense_matrices]
    matrices_and_dimensions = dict(zip(matrices, matrices_dimensions))

    matrices_and_dimensions = {key: value for key, value in sorted(matrices_and_dimensions.items(), key=lambda item: item[1])}
    matrices_paths = [(path + matrix_name) for matrix_name in matrices_and_dimensions.keys()]

    return matrices_and_dimensions, matrices_paths
