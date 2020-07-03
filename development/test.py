import os

if __name__ == '__main__':
    matrices = []
    matrices_locations = []

    for dirname, _, filenames in os.walk('../matrices/mtx'):
        for filename in filenames:
            print(filename)
            matrices_locations.append(os.path.join(dirname, filename))
            matrices.append(filename)

    print(matrices)
