import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from development.path_tools import sorted_matrices_and_paths
from development.power_method import power_method_coo

if __name__ == '__main__':

    file_path = "../matrices/mtx"
    matrices_dict, matrices_locations = sorted_matrices_and_paths(file_path)

    old_stdout = sys.stdout
    log_file = open("../logs/power-method.log", "w")
    sys.stdout = log_file

    delta_time_elapsed_per_matrix = []
    matrices_names = []

    for matrix_location, matrix_name, counter in zip(matrices_locations, matrices_dict.keys(),
                                                     [i for i in range(len(matrices_dict))]):
        matrix_name = matrix_name[:len(matrix_name) - 4]
        matrix = scipy.io.mmread(matrix_location)

        matrix_dimension = len(matrix.todense())
        initial_eigenvector_approximation = np.ones(matrix_dimension)
        tolerance = 10 ** -12
        max_iterations = 20000

        init_power_method = time.time()
        approximated_eigenvalue, approximated_eigenvector = (
            power_method_coo(matrix.tocoo(), initial_eigenvector_approximation, tolerance, max_iterations))
        delta_power_method = time.time() - init_power_method

        delta_time_elapsed_per_matrix.append(delta_power_method)
        matrices_names.append(matrix_name)

        print("-------- MATRIZ: {} --------".format(matrix_name))
        print("Dimensão: {}".format(len(matrix.todense())))
        print("Tempo necessário para encontrar o raio espectral: {:.2f}s\n".format(delta_power_method))

        fig, ax1 = plt.subplots(1, 1)

        ax1.plot(matrices_names, delta_time_elapsed_per_matrix, color="blue")
        ax1.set_ylabel("Time elapsed (s)")
        ax1.set_xlabel("Matrix name")

        plt.setp(ax1.get_xticklabels(), rotation='vertical', fontsize=11)
        plt.suptitle("Power method", y=0.98, fontsize=20)
        plt.subplots_adjust(top=0.85)

        matrix_directory = "../plots/"
        fig.savefig(matrix_directory + "Power-method-" + str(counter + 1) + "matriz" ".png")
        plt.show()

    sys.stdout = old_stdout
    log_file.close()
