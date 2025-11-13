#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

dx = 0.25
sinx = np.sin(x)
cosx = np.cos(x)

fwd_diff = [(sinx[1:] - sin[:-1])] / dx
bkd_diff = [(sinx[1:] - sin[:-1])] / dx
cnt_diff = [(sinx[2:] - sin[:-2])] / (2*dx)

dxs = np.array([2**-n for n in range(20)])
err_fwd, err_cnt = [], []

for dx in dxs:

    x = np.arange(0, 2.5 * np.pi, dx)
    sinx = np.sin(x)

    fwd_diff = [(sinx[1:] - sin[:-1])] / dx
    bkd_diff = [(sinx[1:] - sin[:-1])] / dx
    cnt_diff = [(sinx[2:] - sin[:-2])] / (2*dx)

    err_fwd.append(np.abs(fwd_diff[-1] - np.cos(x[-1])))
    err_cnt.append(np.abs(cnt_diff[-1] - np.cos(x[-2])))

fig, ax = plt.subplots(1,1)
ax.loglog(dxs, err_fwd, '.', label = 'Forward Diff')
ax.loglog(dxs, err_cnt, '.', label = 'Central Diff')
ax.set_xlabel(r'$\Delta x$')







