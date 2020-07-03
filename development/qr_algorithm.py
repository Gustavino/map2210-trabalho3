import numpy as np


def qr_iteration_eigenvalues(A, precision):
    previous_iteration = A[0][0] + 1

    while abs(previous_iteration - A[0][0]) > precision:
        previous_iteration = A[0][0]
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

    return np.diag(A)
