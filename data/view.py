"""
File for generating images to show if the weights are changing.
"""
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def print_w_ts(prob_name, n=400):
    """ Gathers all the w_t's """
    Ws = None
    for i in range(1,n+1):
        fname = "%s/w_%i.csv" % (prob_name, i)
        if(not os.path.isfile(fname)):
            break

        df = pd.read_csv(fname, header=None)
        w_s = df[0].to_numpy()
        if Ws is None:
            Ws = np.zeros((len(w_s), n), dtype=float)
        Ws[:,i-1] = w_s
    return Ws

def difference_in_w_ts(Ws):
    w_t_diff = np.zeros(Ws.shape[1]-1, dtype=float)
    for t in range(len(w_t_diff)):
        w_t = Ws[:,t]
        w_t_next = Ws[:,t+1]
        w_t_diff[t] = la.norm(w_t - w_t_next)
    return w_t_diff

prob_name = "logistic_regression"
Ws = print_w_ts(prob_name)
w_t_diff = difference_in_w_ts(Ws)

# plot
plt.style.use('ggplot')
_, ax = plt.subplots()
ax.plot(w_t_diff)
ax.set(title="Difference in weights for %s" % prob_name, xlabel=r"Iteration $t$", ylabel=r"$\|w_t-w_{t+1}\|_2$")
plt.savefig("%s_wt_diffs.png" % prob_name, dpi=240)
