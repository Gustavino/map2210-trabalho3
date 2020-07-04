import numpy as np

from release.inverse_power_method import inverse_power_method
from release.power_method import infinite_norm, vector_normalization


def power_method_with_inverse_iteration(input_matrix, arbitrary_non_null_vector, tolerance, max_iterations):
    eigenvector = arbitrary_non_null_vector
    greatest_element_index, greatest_element = infinite_norm(eigenvector)
    eigenvector = vector_normalization(eigenvector, greatest_element)

    for current_iteration in range(max_iterations):
        matrix_eigenvector_product = input_matrix.dot(eigenvector)
        greatest_element_index, greatest_element = infinite_norm(matrix_eigenvector_product)

        if abs(greatest_element) < 10 ** -18:
            print("The input matrix has the eigenvalue 0, select a new arbitrary vector and restart.")
            return 0, np.empty(len(input_matrix.todense()))

        error = np.linalg.norm(eigenvector - np.dot(matrix_eigenvector_product, 1 / greatest_element), np.inf)
        eigenvector = np.dot(matrix_eigenvector_product, 1 / greatest_element)

        if error < 10 ** -4:
            eigenvalue, eigenvector = inverse_power_method(np.copy(input_matrix.todense()), eigenvector,
                                                           tolerance, max_iterations)
            return eigenvalue, eigenvector

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")
