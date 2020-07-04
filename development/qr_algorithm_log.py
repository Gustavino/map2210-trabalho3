import scipy.io
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

from development.path_tools import sorted_matrices_and_paths
from development.qr_algorithm import qr_iteration_eigenvalues

if __name__ == '__main__':

    file_path = "../matrices/mtx"
    matrices_dict, matrices_locations = sorted_matrices_and_paths(file_path)

    old_stdout = sys.stdout
    log_file = open("qr-algorithm.log", "w")
    sys.stdout = log_file

    delta_time_elapsed_per_matrix = []
    matrices_dimension = []
    for matrix_location, matrix_name, counter in zip(matrices_locations, matrices_dict.keys(),
                                                     [i for i in range(len(matrices_dict))]):
        matrix_name = matrix_name[:len(matrix_name) - 4]

        qr_start = time.time()
        matrix = np.copy(scipy.io.mmread(matrix_location).todense())
        qr_precision = 10 ** -20
        eigenvalues = qr_iteration_eigenvalues(matrix, qr_precision)
        qr_delta = time.time() - qr_start

        delta_time_elapsed_per_matrix.append(qr_delta)

        print("-------- MATRIZ: {} --------".format(matrix_name))
        print("Quantidade de autovalores: {}".format(len(matrix)))
        print("Tempo necessário para encontrar todos os autovalores: {:.2f}s".format(qr_delta))

        matrices_dimension.append(len(matrix))

        fig, ax1 = plt.subplots(1, 1)

        ax1.plot(matrices_dimension, delta_time_elapsed_per_matrix, color="red")
        ax1.set_ylabel("Time elapsed (s)")
        ax1.set_xlabel("Matrix dimension")
        plt.suptitle("QR Algorithm", y=0.98, fontsize=20)
        plt.subplots_adjust(top=0.85)

        fig.savefig(str(counter + 1) + ".png")
        plt.show()

    sys.stdout = old_stdout
    log_file.close()
