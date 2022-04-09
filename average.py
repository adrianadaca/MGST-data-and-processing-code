import matplotlib.pyplot as plt
import numpy as np


def average(test, i0, t1, t2):

    i1 = i0 + t1
    i2 = i0 + t2
    plt.plot(test.index[i0:-1], test.force_x[i0:-1], color='b')
    plt.plot(test.index[i0:-1], test.force_y[i0:-1], color='k')
    plt.plot(test.index[i1:i2], test.force_x[i1:i2], color='r')
    plt.plot(test.index[i1:i2], test.force_y[i1:i2], color='r')
    #plt.show()

    # division by wheel load
    DP_W = test.force_x[i1:i2] / test.force_y[i1:i2]

    # averages
    DP_avg = np.mean(test.force_x[i1:i2])
    FN_avg = np.mean(test.force_y[i1:i2])
    FZ_avg = np.mean(test.force_z[i1:i2])
    DP_W_avg = np.mean(DP_W)

    return DP_avg, FN_avg, FZ_avg, DP_W_avg, i1, i2
