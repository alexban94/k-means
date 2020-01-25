import numpy as np


def k_means(data, k):
    rows, columns = data.shape
    convergence = False
    prevJ = 0

    # Initialise mu_k randomly as one of the data points in the dataset

    mu_k = np.random.randint(0, rows, size=(k, 1))

    while not convergence:
        # Expectation Step, minimise J for r_nk keeping mu_k fixed
        r_nk = np.zeros((rows, k))
        for i in range(rows):
            mag = np.zeros((1, k))
            for j in range(k):
                distance = (data.iloc[i] - mu_k[j])**2
                mag[0, j] = np.sqrt(np.dot(distance, distance))
            j = np.argmin(mag)
            r_nk[i, j] = 1

        # Maximisation Step
        # Calculate new cluster centres mu_k

        for i in range(k):
            mu_k[i] = np.sum(r_nk[:, i] * data) / np.sum(r_nk[:, i])

        # Convergence check - calculate distortion function
        J = distortion(data, r_nk, mu_k, rows, k)
        if J == prevJ:
            convergence = True
        else:
            prevJ = J
    return mu_k, r_nk

def distortion(data, r_nk, mu_k, rows, k):
    temp = 0
    for i in range(rows):
        for j in range(k):
            temp += r_nk[i, j] * (data.iloc[i] - mu_k[j]) ** 2
    J = np.sqrt(np.dot(temp, temp))
    return J

