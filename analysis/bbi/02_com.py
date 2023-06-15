# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script produces time, autocorrelation, and 
#     average mutual information  plots & other vizualizations 
#     of the COM data (128Hz)
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import scipy as sp
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from src import fun_bbi as bbi
from src import nonlinear_dynamics as nld

from src.params import com_vars, fs, mlag_acf, mlag_ami


# define paths & load data
PATH_DATA = './data/npy/'
PATH_OUT = './output/dat/'

# load data...
data = bbi.import_npz(PATH_DATA)
COM, EEG = data['COM_128'], data['EEG_128']



# COM -------------------------------------------------------
with PdfPages(PATH_OUT + '02_4_com__deg_s.pdf') as pdf:
    # view all center of mass data and PSD
    #   - AP, ML, VT, and RES(AP, ML, VT)

    for ii, xid in enumerate(COM):
        fig, ax = plt.subplots(len(com_vars), 2, 
                               gridspec_kw={'width_ratios': [3, 1]},
                               figsize=(11, 8))
        plt.rcParams.update({'font.size': 8})

        fc = 12
        for j, xvar in enumerate(com_vars):
            # data 
            x, y = COM[xid]['time'], COM[xid][xvar]
            y1 = bbi.lowpass_butter(y, fc=fc, fs=fs, order=4)
            
            # periodogram
            (f, S) = sp.signal.welch(y, fs)

            # plot 
            ax[j, 0].plot(x, y, '-', lw=0.75, label='128Hz')
            ax[j, 0].set_xlabel('time (s)')
            ax[j, 0].set_ylabel(com_vars[xvar])
            ax[j, 0].set_title(xid)
            ax[j, 0].legend(fontsize=6)
            # periodogram
            ax[j, 1].semilogy(f, S, lw=0.75)
            ax[j, 1].set_xlim([-1, 64])
            ax[j, 1].set_xlabel('frequency (Hz)')
            ax[j, 1].set_ylabel('PSD ($\mu V^2/Hz$)')

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_OUT + '02_5_com__acf.pdf') as pdf:
    # autocorrelation plots for all COM data
    #   - AP, ML, VT, and RES(AP, ML, VT)
    #   - includes 128Hz data & 32Hz data (not utilized)

    for ii, xid in enumerate(COM):
        fig = plt.figure(figsize=(8, 3))
        plt.rcParams.update({'font.size': 8})

        counter = 1
        for j, xvar in enumerate(com_vars):
            y = COM[xid][xvar]

            ax = plt.subplot(1, 4, counter)
            # acf 2Hz
            ax.acorr(y, maxlags=mlag_acf, lw=0.75)
            ax.set_xlim([0, mlag_acf])
            ax.set_ylabel('ACF, {}'.format(com_vars[xvar]))
            ax.set_title(xid+ ': {} Hz'.format(fs), loc='left')

            counter += 1

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_OUT + '02_6_com__ami.pdf') as pdf:
    # average mutual information plots for all COM data
    #   - AP, ML, VT, and RES(AP, ML, VT)
    #   - includes 128Hz data & 32Hz data (not utilized)

    for ii, xid in enumerate(COM.keys()):
        fig = plt.figure(figsize=(8, 3))
        plt.rcParams.update({'font.size': 8})

        counter = 1
        for j, xvar in enumerate(com_vars):
            x = COM[xid]['time']
            y = COM[xid][xvar]
            # x1, y1 = bbi.downsample([x, y], fs, fsn)

            mi = nld.ami(y, maxlags=int(mlag_ami))
            # mi1 = nld.ami(y1, maxlags=int(maxlags/4))

            ax = plt.subplot(1, 4, counter)
            ax.plot(mi[0], mi[1], lw=0.75, label='128Hz')
            # ax.plot(np.arange(1, 127, 2), mi1[1], lw=0.75, label='32Hz')
            ax.axvline(16, ls=':', lw=0.25, alpha=0.75)
            ax.axvline(32, ls=':', lw=0.25, alpha=0.75)
            ax.set_xlim(0, mlag_ami)
            # ax.set_xticks(ticks=[0, 64, 128], labels=[0, 0.5, 1])
            ax.set_xlabel('lags')
            ax.set_ylabel('AMI ({})'.format(com_vars[xvar]))
            ax.set_title(xid+ ': {} Hz'.format(fs), loc='left')
            ax.legend(fontsize=6)

            counter += 1

        plt.tight_layout()
        pdf.savefig()
        plt.close()


# end 
