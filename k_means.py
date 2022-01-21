import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

def k_means(data, k, max_iter):
    rows, columns = data.shape
    convergence = False
    prevJ = 0



    # Initialise mu_k as randomly selected samples in the dataset
    mu_k_index = np.random.randint(0, rows, size=(k, 1))
    mu_k = np.zeros([k, columns])
    for i in range(k):
        mu_k[i] = data[mu_k_index[i]][:]

    # Store initial cluster centers
    mu_viz = [mu_k]
    r_viz =  []
    iteration = 0
    # Repeat until convergence - no further change in J between iterations.
    while iteration <= max_iter and not convergence:
        # Expectation Step, minimise J for r_nk keeping mu_k fixed
        r_nk = np.zeros((rows, k))
        for i in range(rows):
            mag = np.zeros((1, k))
            for j in range(k):
                # Index ith row, for cluster centre j.
                cur = data[i,:]
                mu = mu_k[j]

                # Calculate Euclidean distance
                distance = (cur - mu) **2
                mag[0, j] = np.sqrt(np.dot(distance, np.transpose(distance)))
            # Entry i is assigned to the cluster that is closest.
            j = np.argmin(mag)
            r_nk[i, j] = 1

        # Maximisation Step
        # Calculate new cluster centres mu_k

        for i in range(k):
            temp = np.tile(r_nk[:,i], (columns,1))
            tp = np.transpose(temp)
            mu_k[i] = np.sum(tp * data) / np.sum(r_nk[:, i])

        # Convergence check - calculate distortion function
        J = distortion(data, r_nk, mu_k, rows, k)
        if J == prevJ:
            convergence = True
        else:
            prevJ = J

        iteration += 1
        print(iteration)

        # r_nk and mu_k assigned for this iteration, store data for visualization.
        # For ease, animation is done once the K-Means process is finished.
        # Update cluster centres, mu_viz
        mu_viz.append(mu_k)
        # Update cluster assignments, r_viz
        r_viz.append(update_assignments(data, r_nk))

    # Once converged, can run the animation.
    # Convert list of arrays to 2d array
    r_viz = np.concatenate(r_viz, axis = 1)

    # Define the figure, plot, animation
    # Create figure
    fig = plt.figure(figsize=(7, 7), facecolor='black')
    ax = plt.subplot(111, frameon=False)
    # Subplot with no frame
    # Interval is the delay between frame updates, frame is the iterable for the animation
    # (i.e, the number of iterations of K-Means).
    animate = animation.FuncAnimation(fig, update_visualisation, frames=range(np.size(r_viz,1)),
                                      fargs=(data, r_viz, mu_viz), interval=1000)
    plt.show()

    return mu_k, r_nk


def update_visualisation(frame, *args):
    # *args is the data, r_nk, mu_k values. Indexed by the frame for the current stage of the animation.
    # Unpack the arguments using frame.
    data = args[0]
    r_nk = args[1][:,frame]
    mu_k = args[2][frame]

    # Update visuals

    data = np.hstack((data[:,0], data[:,1], r_nk))


    plt.cla()


    plt.scatter(data[:,0], data[:,1], c=data[:,2], alpha=0.8, marker='x')
    plt.scatter(mu_k[:,0], mu_k[:,1], c=[1,2,3])

    return



def update_assignments(data, r_nk):
    # Convert r_nk into a 'class' for colour separation
    rows, col  = data.shape
    labels = np.empty([rows, 1])
    for i in range(rows):
        if r_nk[i,1] == 1:
            labels[i,:] = 1
        elif r_nk[i,2] == 1:
            labels[i,:] = 2
        else:
            labels[i,:] = 3
    return labels


def distortion(data, r_nk, mu_k, rows, k):
    temp = 0
    for i in range(rows):
        for j in range(k):
            temp += r_nk[i, j] * (data[i,:] - mu_k[j]) ** 2
    J = np.sqrt(np.dot(temp, temp))
    return J

