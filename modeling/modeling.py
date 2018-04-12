from scipy.optimize import fmin, fmin_cg, fmin_bfgs, basinhopping, differential_evolution
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
   
from models import minsky_pageable_bw




df = pd.read_csv("/home/pearson/repos/ms-thesis/data/plot2/m2_pageable.csv")
data = df.as_matrix()

xdata = data[:, 0] * 1024 * 1024
ydata = data[:, 13] * 1024 * 1024
interp = interp1d(xdata, ydata)
ax = plt.plot(xdata, ydata, 'o', label='data')
ax = plt.plot(xdata, interp(xdata), '-', label='interp')
plt.xscale('log')

def callback(xk, convergence):
    print list(xk), convergence
    return False

def objective(params):


    overhead  = params[0]
    nvlink_bw  = params[1]
    l1_bw  = params[2]
    l1_size  = params[3]
    l2_bw = params[4]
    l2_size  = params[5]
    l3_bw = params[6]
    l3_size  = params[7]
    m_bw = params[8]

    sad = 0
    for bytes in xdata:
        model_bw = minsky_pageable_bw(bytes, overhead, nvlink_bw, l1_bw, l1_size, l2_bw, l2_size, l3_bw, l3_size, m_bw)
        actual_bw = interp(bytes)
        sad += abs(model_bw - actual_bw)
    return sad

# x0 = [1e-4, 1e10, 1e12, 1e3, 1e12, 1e5, 1e12, 1e6, 1e10]
x0 = [1.7e-05, 1.2e10, 3e13, 5e6, 8e9, 3e7, 2e10, 1e8, 1.15e10]
# x0 = [1e-4, 1e13, 1e13, 1e3, 1e12, 1e5, 1e12, 1e6, 1e10]
initdata = minsky_pageable_bw(xdata, *x0)
plt.plot(xdata, initdata, label='inital')
# plt.show()

bounds = [(0.0,1.0),(1e9, 1e12),(1e8, 1e14),(1e2, 1e6),(1e8, 1e14),(1e3, 1e6),(1e8, 1e14),(1e6, 1e8),(1e8, 1e14)]
# xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag, allvecs = fmin_bfgs(objective, x0, maxiter=50, full_output=True, retall=True)
# res = basinhopping(objective, x0, niter=50, callback=callback)
res = differential_evolution(objective, bounds, callback=callback, strategy="best1exp")
print res
print res.x
# for v in allvecs:
#     print v
mdata = minsky_pageable_bw(xdata, *res.x)
print str(res.x[0]) + "s overhead"
print str(res.x[1] / 1e9) + " GB/s link bandwidth"
print str(res.x[2] / 1e9) + " GB/s l1 bandwidth"
print str(res.x[3] / 1e6) + " MB cache"
print str(res.x[4] / 1e9) + " GB/s mem bandwidth"

plt.plot(xdata, mdata)
plt.show()


