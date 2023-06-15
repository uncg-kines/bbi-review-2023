#
# nate berry
# 2023-04-01
#
#
# ---------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Cross
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius


def vdp(t, state, mu):
    x, y = state
    return [y, mu*(1 - x**2)*y - x]



# model parameters ----------------------------
METHOD = 'LSODA'
t0, tf, N = 0, 150, 1000
t = np.linspace(t0, tf, N)
# state 
state = [1., 0.]
# parameters
mu_list = [0., 1., 2.2, 2.2, 4.0]
xylim = [-7, 7]


# plots ---------------------------------------
fig, ax = plt.subplots(5, 3,
                       gridspec_kw={'width_ratios': [1, 3, 1]},
                       figsize=(5, 5.6))
plt.rcParams.update({'font.size': 8})
for ii, mu in enumerate(mu_list):
    # solve vdp 
    sol0 = solve_ivp(vdp, [t0, tf], state, method=METHOD,
                     t_eval=t, args=(mu,))
    # mu = 2.5
    sol1 = solve_ivp(vdp, [t0, tf], state, method=METHOD,
                     t_eval=t, args=(2.5,))

    if ii == 3:
        sol0.y[0] = -1*sol0.y[0]

    # calculate Takens
    y0 = TimeSeries(sol0.y[0], embedding_dimension=3, time_delay=13)
    y1 = TimeSeries(sol1.y[0], embedding_dimension=3, time_delay=13)
    ts = (y0, y1)  # definte crossed time series
    # radius and theiler
    r, TC = 0.4, 0

    settings = Settings(ts,
                        analysis_type=Cross,
                        neighbourhood=FixedRadius(r),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=TC)
    computation = RQAComputation.create(settings, verbose=False)
    rqa_result = computation.run()
    plot_computation = RPComputation.create(settings)
    plot_result = plot_computation.run()
    im = plot_result.recurrence_matrix_reverse

    # plot state space
    ax[ii, 0].plot(sol0.y[0], sol0.y[1], lw=0.6)
    ax[ii, 0].plot(sol1.y[0], sol1.y[1], lw=0.6, ls='--')
    ax[ii, 0].set_xlim(xylim)
    ax[ii, 0].set_ylim(xylim)
    ax[ii, 0].set_xticks([])
    ax[ii, 0].set_yticks([])
    ax[ii, 0].set_xlabel('x')
    ax[ii, 0].set_ylabel('y')
    ax[ii, 0].set_aspect(1.)
    # plot time
    ax[ii, 1].plot(sol0.t, sol0.y[0], lw=0.7)
    ax[ii, 1].plot(sol1.t, sol1.y[0], lw=0.7, ls='--')
    ax[ii, 1].set_ylim(-5, 5)
    ax[ii, 1].set_xlabel('time')
    ax[ii, 1].set_ylabel('x')
    # plot recurrence
    ax[ii, 2].imshow(im, cmap='binary')
    # ax[ii, 2].set_xticks(ticks=np.arange(0, N+1, N/2),
    #                      labels=np.arange(0, int((N)/10)+1, int(N/2/10)))
    # ax[ii, 2].set_yticks(ticks=np.arange(N+1, 0, -(N/2)),
    #                      labels=np.arange(0, int((N)/10)+1, int(N/2/10)))
    ax[ii, 2].set_xticks([])
    ax[ii, 2].set_yticks([])
    ax[ii, 2].set_aspect('equal')
    ax[ii, 2].set_xlabel('$x_{1}$')
    ax[ii, 2].set_ylabel('$x_{2}$')

plt.tight_layout()
plt.savefig('./output/sims/fig2.jpg', dpi=500)
plt.show()

# end
