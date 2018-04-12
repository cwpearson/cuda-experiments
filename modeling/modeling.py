from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
   

def pinned_bw(x, c, b_max):

    def time(x, c, b_max):
        return c + x / b_max

    return x / time(x, c, b_max)


def pageable_bw(x, c, b_max, b_cache, cache_size, b_mem):

    def copy_time(x, b_cache, cache_size, b_mem):
        if x < cache_size:
            copy_time = x / b_cache
        else:
            copy_time = x / b_mem
        return copy_time

    def time(x, c, b_max, b_cache, cache_size,  b_mem):
        return c + x / b_max + np.vectorize(copy_time)(x, b_cache, cache_size, b_mem)

    return x / time(x, c, b_max, b_cache, cache_size, b_mem)




df = pd.read_csv("/home/pearson/repos/ms-thesis/data/plot2/m2_pageable.csv")
data = df.as_matrix()

xdata = data[:, 0]
ydata = data[:, 1]
interp = interp1d(xdata, ydata)
ax = plt.plot(xdata, ydata, 'o', label='data')
ax = plt.plot(xdata, interp(xdata), '-', label='interp')
plt.xscale('log')


def objective(params):

    c = params[0]
    b_max = params[1]
    b_cache = params[2]
    cache_size = params[3]
    b_mem = params[4]

    sad = 0
    for x in xdata:
        sad += abs(pageable_bw(x, c, b_max, b_cache, cache_size, b_mem) - interp(x))
    return sad

x0 = [1e-4, 3e4, 1e6, 16, 2e4]
sigma = np.ones(xdata.shape) /1e3
xopt, fopt, iter, funcalls, warnflag, allvecs = fmin(objective, x0, maxiter=50, full_output=True, retall=True)
print xopt
for v in allvecs:
    print v
mdata = pageable_bw(xdata, *xopt)
# print str(xopt[0]) + "s overhead"
# print str(xopt[1] / 1024.0 ) + "GB/s link bandwidth"
# print str(xopt[2] / 1024.0 ) + "GB/s cache bandwidth"
# print str(xopt[3]          ) + "MB cache"
# print str(xopt[4] / 1024.0 ) + "GB/s mem bandwidth"

plt.plot(xdata, mdata)
plt.show()


def objective(x):

    def model(count, theta):
        return theta[0] + count / theta[1]

    sad = 0.0
    for x in range(10):
        sad += abs(actual(count) - model(count))
    x[0]