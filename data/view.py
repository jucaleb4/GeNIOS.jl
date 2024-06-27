"""
File for generating images to show if the weights are changing.
"""
import os
import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def get_var_ts(prob_name, var_name, n=100):
    """ Gathers all the w_t's """
    Ws = None
    ct = 1
    while ct < n+1:
        fname = "%s/%s_%i.csv" % (prob_name, var_name, ct)
        # no more files left
        if(not os.path.isfile(fname)):
            break

        df = pd.read_csv(fname, header=None)
        w_s = df[0].to_numpy()
        if Ws is None:
            Ws = np.zeros((len(w_s), n), dtype=float)
        Ws[:,ct-1] = w_s
        ct += 1
    return Ws[:,:ct].copy()

def difference_in_var_ts(Xs):
    x_t_diff = np.zeros(Xs.shape[1]-1, dtype=float)
    for t in range(len(x_t_diff)):
        x_t = Xs[:,t]
        x_t_next = Xs[:,t+1]
        x_t_diff[t] = la.norm(x_t - x_t_next)
    return x_t_diff

folder_name = sys.argv[1]
prob_name = sys.argv[2]
prob_name = os.path.join(folder_name, "logistic_regression_%s" % prob_name)
Ws = get_var_ts(prob_name, 'w', 200)
Xs = get_var_ts(prob_name, 'x', 200)
Zs = get_var_ts(prob_name, 'z', 200)

w_t_diff = difference_in_var_ts(Ws)
x_t_diff = difference_in_var_ts(Xs)
z_t_diff = difference_in_var_ts(Zs)

# plot
plt.style.use('ggplot')
fig, axes = plt.subplots(ncols=1, nrows=3)
fig.set_size_inches(6,6)
var_diff = [w_t_diff, x_t_diff, z_t_diff]
var_name = ['w', 'x', 'z']
for i in range(3):
    var = var_name[i]
    axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[i].plot(var_diff[i])
    axes[i].set(
        xlabel=r"Iteration $t$", 
        ylabel=r"$\|%s_t-%s_{t+1}\|_2$" % (var, var)
    )

plt.suptitle("Variable diffs for %s" % prob_name)
plt.tight_layout()
plt.show()
# plt.savefig("%s_wt_diffs.png" % prob_name, dpi=240)
