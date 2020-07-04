import numpy as np


def infinite_norm(vector):
    max_element_index = np.argmax(vector)
    min_element_index = np.argmin(vector)

    if abs(vector[min_element_index]) == abs(vector[max_element_index]):
        if min_element_index < max_element_index:
            return min_element_index, vector[min_element_index]
        return max_element_index, vector[max_element_index]

    if abs(vector[min_element_index]) > abs(vector[max_element_index]):
        return min_element_index, vector[min_element_index]
    return max_element_index, vector[max_element_index]


def vector_normalization(vector, element):
    return np.dot(1 / element, vector)


def power_method_dense(input_matrix, arbitrary_non_null_vector, tolerance, max_iterations):
    greatest_element_index, greatest_element = infinite_norm(arbitrary_non_null_vector)
    arbitrary_non_null_vector = vector_normalization(arbitrary_non_null_vector, greatest_element)

    for current_iteration in range(max_iterations):
        next_iteration_eigenvector = np.dot(input_matrix, arbitrary_non_null_vector)
        eigenvalue = next_iteration_eigenvector[greatest_element_index]
        greatest_element_index, greatest_element = infinite_norm(next_iteration_eigenvector)

        if abs(greatest_element) < 10 ** -18:
            print("The input matrix has the eigenvalue 0, select a new arbitrary vector and restart.")
            return 0, np.empty(len(input_matrix.todense()))

        error = np.linalg.norm(arbitrary_non_null_vector - np.dot(next_iteration_eigenvector, 1 / greatest_element), np.inf)
        arbitrary_non_null_vector = np.dot(next_iteration_eigenvector, 1 / greatest_element)  # Eigenvector

        if error < tolerance:
            print("The procedure was successful in {} iterations".format(current_iteration + 1))
            eigenvector = arbitrary_non_null_vector
            return eigenvalue, eigenvector

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")


def power_method_coo(input_matrix, arbitrary_non_null_vector, tolerance, max_iterations):
    eigenvector = arbitrary_non_null_vector
    greatest_element_index, greatest_element = infinite_norm(eigenvector)
    eigenvector = vector_normalization(eigenvector, greatest_element)

    for current_iteration in range(max_iterations):
        next_iteration_eigenvector = input_matrix.dot(eigenvector)
        eigenvalue = next_iteration_eigenvector[greatest_element_index]
        greatest_element_index, greatest_element = infinite_norm(next_iteration_eigenvector)

        if abs(greatest_element) < 10 ** -18:
            print("The input matrix has the eigenvalue 0, select a new arbitrary vector and restart.")
            return 0, np.empty(len(input_matrix.todense()))

        error = np.linalg.norm(eigenvector - np.dot(next_iteration_eigenvector, 1 / greatest_element), np.inf)
        eigenvector = np.dot(next_iteration_eigenvector, 1 / greatest_element)  # Eigenvector

        if error < tolerance:
            return eigenvalue, eigenvector

    raise Exception("Maximum number of iterations exceeded. The procedure was unsuccessful.")
