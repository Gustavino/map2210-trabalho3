import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from development.path_tools import sorted_matrices_and_paths
from development.power_and_inverse import power_method_with_inverse_iteration

if __name__ == '__main__':
    file_path = "../matrices/mtx"
    matrices_dict, matrices_locations = sorted_matrices_and_paths(file_path)

    old_stdout = sys.stdout
    log_file = open("../logs/power-method-with-inverse.log", "w")
    sys.stdout = log_file

    delta_time_elapsed_per_matrix = []
    matrices_dimensions = []

    for matrix_location, matrix_name, counter in zip(matrices_locations, matrices_dict.keys(),
                                                     [i for i in range(len(matrices_dict))]):
        matrix_name = matrix_name[:len(matrix_name) - 4]

        matrix = scipy.io.mmread(matrix_location)
        initial_eigenvector_approximation = np.ones(len(matrix.todense()))
        tolerance = 10 ** -12
        max_iterations = 50000

        init_power_and_inverse = time.time()
        approximated_eigenvalue, approximated_eigenvector = (
            power_method_with_inverse_iteration(matrix.tocoo(), initial_eigenvector_approximation, tolerance,
                                                max_iterations))
        delta_power_and_inverse = time.time() - init_power_and_inverse

        delta_time_elapsed_per_matrix.append(delta_power_and_inverse)
        matrices_dimensions.append(len(matrix.todense()))

        print("-------- MATRIZ: {} --------".format(matrix_name))
        print("Dimensão: {}".format(len(matrix.todense())))
        print("Tempo necessário para encontrar o raio espectral utilizando o método da potência inversa como "
              "otimizador: {:.2f}s\n".format(delta_power_and_inverse))

        fig, ax1 = plt.subplots(1, 1)

        ax1.plot(matrices_dimensions, delta_time_elapsed_per_matrix, color="blue")
        ax1.set_ylabel("Time elapsed (s)")
        ax1.set_xlabel("Matrix dimension")
        plt.setp(ax1.get_xticklabels(), rotation='vertical', fontsize=11)
        plt.suptitle("Power method with inverse iteration", y=0.98, fontsize=20)
        plt.subplots_adjust(top=0.85)

        fig.savefig("Power-method-with-inverse-" + str(counter + 1) + "matrix" ".png")
        plt.show()

    sys.stdout = old_stdout
    log_file.close()
