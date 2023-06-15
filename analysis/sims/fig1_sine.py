#
# nate berry
# 2023-04-01
#
#
# ---------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Cross
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.image_generator import ImageGenerator


# define functions
def f0(t):
    y = np.sin(t)
    return (y)


def f1(t):
    y = np.sin(t) * 2
    return (y)


def f2(t):
    y = np.sin(t) * 2 + 1
    return (y)


def f3(t):
    y = np.sin(t) + 0.01 * t + 2e-05 * t**2
    return (y)


def f4(t):
    y = np.cos(t)
    return (y)


# list functions and eqs
funlist = [f0, f1, f2, f3, f4]
funlab = ['sin(t)',
          '2sin(t)',
          '2sin(t) + 1',
          '2e5$t^2$+sin(t)+0.1t',
          'cos(t)']


N = 1500
eps = np.random.normal(0, 0.05, N)
eps1 = np.random.normal(0, 0.05, N)
clist = plt.rcParams['axes.prop_cycle'].by_key()['color']

# plot
fig, ax = plt.subplots(5, 3,
                       gridspec_kw={'width_ratios': [1, 3, 1]},
                       figsize=(5.5, 5.6))
plt.rcParams.update({'font.size': 8})
for ii, fn in enumerate(funlist):
    t = np.linspace(0, 150, N)  # time
    fn0 = funlist[0]
    y0, y1 = fn0(t), fn(t)

    # calculate ts for crqa
    y00 = TimeSeries(y1, embedding_dimension=3, time_delay=13)
    y11 = TimeSeries(y0, embedding_dimension=3, time_delay=13)
    ts = (y00, y11)  # definte crossed time series
    r, TC = [0.5]*5, 0
    settings = Settings(ts,
                        analysis_type=Cross,
                        neighbourhood=FixedRadius(r[ii]),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=TC)
    computation = RQAComputation.create(settings, verbose=False)
    rqa_result = computation.run()
    plot_computation = RPComputation.create(settings)
    plot_result = plot_computation.run()
    im = plot_result.recurrence_matrix_reverse

    # plot state space
    ax[ii, 0].set_aspect('equal', adjustable='box')
    ax[ii, 0].plot(y0[:-12], y0[12:], lw=0.5, alpha=0.6, label=funlab[0])
    ax[ii, 0].plot(y1[:-12], y1[12:], lw=0.5, ls='--', alpha=0.6, label=funlab[ii])
    ax[ii, 0].set_xlim(-2.5, 3.5)
    ax[ii, 0].set_ylim(-2.5, 3.5)
    ax[ii, 0].set_xticks([])
    ax[ii, 0].set_yticks([])
    ax[ii, 0].set_xlabel('f(t)')
    ax[ii, 0].set_ylabel(r'f(t)+$\tau$')
    # plot time series
    ax[ii, 1].plot(t, y0, lw=0.5, alpha=0.75, label=funlab[0])
    ax[ii, 1].plot(t, y1, lw=0.5, alpha=0.75, ls='--', label=funlab[ii])
    ax[ii, 1].set_ylim(-5, 3.5)
    ax[ii, 1].set_yticks(ticks=np.arange(-2, 3, 2))
    ax[ii, 1].set_xlabel('time')
    ax[ii, 1].legend(loc='lower left', fontsize=5, ncol=2, frameon=False)
    # plot recurrence
    ax[ii, 2].imshow(im, cmap='binary')
    # ax[ii, 2].set_xticks(ticks=np.arange(0, N+1, N/2),
    #                      labels=np.arange(0, int((N)/10)+1, int(N/2/10)))
    # ax[ii, 2].set_yticks(ticks=np.arange(N+1, 0, -(N/2)),
    #                      labels=np.arange(0, int((N)/10)+1, int(N/2/10)))
    ax[ii, 2].set_xticks([])
    ax[ii, 2].set_yticks([])
    ax[ii, 2].set_xlabel(funlab[ii])
    ax[ii, 2].set_ylabel(funlab[0])

plt.tight_layout()
plt.savefig('./output/sims/fig1.jpg', dpi=500)
plt.show()

# end
