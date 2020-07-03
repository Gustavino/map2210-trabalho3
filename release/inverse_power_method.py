import numpy as np
from numpy.linalg import LinAlgError

from release.power_method import infinite_norm, vector_normalization


def inverse_power_method(input_matrix, eigenvector_approximation, tolerance, max_iterations):
    input_matrix_dimension = len(input_matrix)

    transpose_product = np.dot(eigenvector_approximation, input_matrix)
    rayleigh_quotient = np.dot(transpose_product, eigenvector_approximation) / np.dot(eigenvector_approximation,
                                                                                      eigenvector_approximation)

    greatest_element_index, greatest_element = infinite_norm(eigenvector_approximation)
    eigenvector_approximation = vector_normalization(eigenvector_approximation, greatest_element)

    for current_iteration in range(max_iterations):
        rayleigh_identity = np.dot(rayleigh_quotient, np.identity(input_matrix_dimension))
        auxiliary_matrix = np.subtract(input_matrix, rayleigh_identity)

        try:
            y = np.linalg.solve(auxiliary_matrix, eigenvector_approximation)
        except LinAlgError:
            print("y does not have unique solution, so {} is an eigenvalue".format(rayleigh_quotient))
            return rayleigh_quotient, np.empty(0)

        eigenvalue = y[greatest_element_index]
        greatest_element_index, greatest_element = infinite_norm(y)

        y_quotient = np.dot(y, 1 / greatest_element)
        error_index, error = infinite_norm(np.subtract(eigenvector_approximation, y_quotient))
        eigenvector_approximation = np.dot(y, 1 / greatest_element)

        if error < tolerance:
            eigenvalue = 1 / eigenvalue + rayleigh_quotient
            return eigenvalue, eigenvector_approximation

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")