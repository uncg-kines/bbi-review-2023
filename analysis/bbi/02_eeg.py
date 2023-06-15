# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script produces time, autocorrelation, and 
#     average mutual information  plots & other vizualizations 
#     of the EEG data (128Hz)
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from src import fun_bbi as bbi
from src import nonlinear_dynamics as nld

from src.params import eeg_bands, eeg_channels, fs, mlag_acf, mlag_ami


# define paths
PATH_DATA = './data/npy/'
PATH_OUT = './output/dat/'

# load data...
data = bbi.import_npz(PATH_DATA)
COM, EEG = data['COM_128'], data['EEG_128']



# EEG ------------------------------------------------------
with PdfPages(PATH_OUT + '02_1_eeg__bands.pdf') as pdf:
    # view all of the raw EEG data
    #   - all bands and all channels
    #   - includes 128Hz and 32Hz (not utilized)

    for xid in EEG:
        for band in eeg_bands:
            fig = plt.figure(figsize=(11.5, 8))
            plt.rcParams.update({'font.size': 7})

            for channel in range(32):
                # band and time for EEG (2 and 32 Hz)
                x, y = EEG[xid]['time'], EEG[xid][band][:, channel]

                ax = plt.subplot(4, 8, channel+1)
                ax.plot(x, y, '-', lw=0.5, label='128Hz')
                if channel>=24: ax.set_xlabel('time (s)')
                ax.set_ylabel('{}, $\mu V^2/Hz$'.format(band))
                ax.set_title(xid + ', ch: {}'.format(channel+1))
                ax.legend(fontsize=6)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


with PdfPages(PATH_OUT + '02_2_eeg__acf.pdf') as pdf:
    # autocorrelation plots for all EEG data
    #   - all bands and all channels
    #   - only includes 128Hz data

    for xid in EEG:
        fig, ax = plt.subplots(4, 6, figsize=(11.5, 8.5), sharex=True, sharey=True)
        plt.rcParams.update({'font.size': 7})

        for k, lobe in enumerate(eeg_channels):
            if k==1: k = 3
            
            for ii, band in enumerate(eeg_bands):
                for j, channel in enumerate(eeg_channels[lobe]):

                    x = EEG[xid]['time']
                    y = EEG[xid][band][:, channel-1]

                    ax[ii, j+k].acorr(y, maxlags=mlag_acf, lw=0.75)
                    ax[ii, j+k].set_xlim([0, mlag_acf])
                    ax[ii, j+k].set_ylabel('ACF, {}'.format(band))
                    ax[ii, j+k].set_title(xid + ' {} Hz'.format(fs), loc='left')
                    ax[ii, j+k].set_title('{}: {}'.format(lobe, eeg_channels[lobe][channel]), loc='right')

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_OUT + '02_3_eeg__ami.pdf') as pdf:
    # average mutual information plots for all EEG data 
    #   - all bands and all channels
    #   - only includes 128Hz data

    for xid in EEG:
        fig, ax = plt.subplots(4, 6, figsize=(11.5, 8.5), sharex=True, sharey=True)
        plt.rcParams.update({'font.size': 7})

        for k, lobe in enumerate(eeg_channels):
            if k==1: k = 3
            
            for ii, band in enumerate(eeg_bands):
                for j, channel in enumerate(eeg_channels[lobe]):

                    x = EEG[xid]['time']
                    y = EEG[xid][band][:, channel-1]

                    # nutual information 
                    mi = nld.ami(y, maxlags=mlag_ami)

                    ax[ii, j+k].plot(mi[0], mi[1], lw=0.75)
                    ax[ii, j+k].axvline(16, ls=':', lw=0.25, alpha=0.75)
                    ax[ii, j+k].axvline(32, ls=':', lw=0.25, alpha=0.75)
                    ax[ii, j+k].set_ylabel('AMI, {}'.format(band))
                    ax[ii, j+k].set_title(xid + ' {} Hz'.format(fs), loc='left')
                    ax[ii, j+k].set_title('ch: {}'.format(eeg_channels[lobe][channel]), loc='right')

        plt.tight_layout()
        pdf.savefig()
        plt.close()


# end 
