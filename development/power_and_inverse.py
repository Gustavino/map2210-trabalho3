import numpy as np

from development.inverse_power_method import inverse_power_method
from development.power_method import vector_normalization, infinite_norm


def power_method_with_inverse_iteration(input_matrix, arbitrary_non_null_vector, tolerance, max_iterations):
    greatest_element_index, greatest_element = infinite_norm(arbitrary_non_null_vector)
    arbitrary_non_null_vector = vector_normalization(arbitrary_non_null_vector, greatest_element)

    for current_iteration in range(max_iterations):

        y = input_matrix.dot(arbitrary_non_null_vector)
        greatest_element_index, greatest_element = infinite_norm(y)

        if abs(greatest_element) < 10 ** -18:
            print("The input matrix has the eigenvalue 0, select a new arbitrary vector and restart.")
            return 0, np.empty(len(input_matrix.todense()))

        error = np.linalg.norm(arbitrary_non_null_vector - np.dot(y, 1 / greatest_element), np.inf)
        arbitrary_non_null_vector = np.dot(y, 1 / greatest_element)

        if error < 10 ** -4:
            eigenvalue, eigenvector = inverse_power_method(np.copy(input_matrix.todense()), arbitrary_non_null_vector,
                                                           tolerance, max_iterations)
            # print("The procedure was successful in {} iterations".format(current_iteration + 1))
            # print("The procedure was successful")
            eigenvector = arbitrary_non_null_vector
            return eigenvalue, eigenvector

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")
