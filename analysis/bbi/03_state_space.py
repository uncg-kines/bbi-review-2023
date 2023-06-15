# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script produces plots of the time series, the
#     false nearest neighbors data from <03_fnn.py>, and
#     the state space reconstructions for the COM and EEG 
#     data (band-wise & aggregated)
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from src import fun_bbi as bbi
from src import nonlinear_dynamics as nld

from src.params import IDX, eeg_bands, eeg_channels, com_vars, max_m_fnn, \
m_eeg, m_com, tau_eeg, tau_com, trange


# define paths & file names
PATH_DATA = './data/npy/'
PATH_DATA_NLD = './data/nld/'
PATH_FIGS = './output/dat/'
# fnn data 
FILE_FNN = '03_fnn.xlsx'

# load data... 
data = bbi.import_npz(PATH_DATA)
COM, EEG = data['COM_128'], data['EEG_128']
EEG_AGG = bbi.agg_bands(EEG, eeg_bands, eeg_channels)



with PdfPages(PATH_FIGS + '03_1_bbi__time_series.pdf') as pdf:
    # plot the channel-wise EEG data with COM data
    #   - includes all EEG channels & AP, ML, VT, RES(AP, ML, VT) from COM

    for xid in EEG:
        for lobe in eeg_channels:
            fig, ax = plt.subplots(len(eeg_bands), len(com_vars),
                                   figsize=(11.5, 7), sharex=True, sharey=True)
            for ii, band in enumerate(eeg_bands):
                for j, xvar in enumerate(com_vars.keys()):
                    # com data 
                    tcom = COM[xid]['time']
                    com = bbi.normalize(COM[xid][xvar])

                    for ic, channel in enumerate(eeg_channels[lobe]):
                        # eeg data
                        teeg = EEG[xid]['time'].reshape(-1)
                        eeg = bbi.normalize(EEG[xid][band][:, channel-1])
                        # teeg, eeg = bbi.downsample([teeg, eeg], fs, fsn)
                        
                        if ic==0: ax[ii, j].plot(tcom[:-128], com[:-128], lw=0.5,
                                                 alpha=0.5, label=com_vars[xvar])
                        ax[ii, j].plot(teeg[:-128], eeg[:-128], lw=0.5, alpha=0.75,
                                       label=eeg_channels[lobe][channel])
                        if ii==3: ax[ii, j].set_xlabel('time (s)')
                        ax[ii, j].set_title(band, fontsize=8)
                        ax[ii, j].set_title(xid, loc='left', fontsize=6)
                        ax[ii, j].set_title(lobe, loc='right', fontsize=6)
                        ax[ii, j].legend(fontsize=6)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


with PdfPages(PATH_FIGS + '03_2_bbi__time_series_agg.pdf') as pdf:
    # plot the aggregated EEG data with COM data
    #   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
    #   - COM data [AP, ML, VT, RES(AP, ML, VT)]

    for xid in EEG_AGG:
        fig, ax = plt.subplots(len(eeg_bands), len(com_vars),
                               figsize=(11.5, 7), sharex=True, sharey=True)
        for k, lobe in enumerate(eeg_channels):
            for ii, band in enumerate(eeg_bands):
                for j, xvar in enumerate(com_vars.keys()):
                    # com data 
                    tcom = COM[xid]['time']
                    com = bbi.normalize(COM[xid][xvar])
                    # tcom, com = bbi.downsample([tcom, com], fs, fsn)
                    
                    # eeg data
                    teeg = EEG_AGG[xid][lobe]['time'].reshape(-1)
                    eeg = bbi.normalize(EEG_AGG[xid][lobe][band])
                    # teeg, eeg = bbi.downsample([teeg, eeg])
                    
                    if k==0: ax[ii, j].plot(tcom[:-128], com[:-128], lw=0.5, alpha=0.5, label=com_vars[xvar])
                    ax[ii, j].plot(teeg[:-128], eeg[:-128], lw=0.6, alpha=0.8, label=lobe)
                    if ii==3: ax[ii, j].set_xlabel('time (s)')
                    ax[ii, j].set_title(band, fontsize=8)
                    ax[ii, j].set_title(xid, loc='left', fontsize=6)
                    ax[ii, j].legend(fontsize=6)

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_FIGS + '03_3_bbi__fnn_agg.pdf') as pdf:
    # false nearest neighbors for aggregated EEG and COM data 
    #   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
    #   - COM data [AP, ML, VT, RES(AP, ML, VT)]

    fnn = pd.read_excel(PATH_DATA_NLD + FILE_FNN)
    edims = np.arange(1, max_m_fnn+1)

    for xid in IDX:
        fig, ax = plt.subplots(len(eeg_bands), 2,
                               figsize=(4, 8), sharex=True, sharey=True)

        for ii, band in enumerate(eeg_bands):
            for lobe in eeg_channels:
                df = fnn.loc[(fnn.id==xid) & (fnn.lobe==lobe) & (fnn.band==band), :]

                # plot 
                ax[ii, 0].plot(edims, df.iloc[:,4:].T, alpha=0.75, label=lobe)
                ax[ii, 0].set_xlabel('$\it{m}$')
                ax[ii, 0].set_xticks(np.arange(2, 10, 2))
                ax[ii, 0].set_title(band, fontsize=8)
                ax[ii, 0].set_title(xid, loc='left', fontsize=6)
                ax[ii, 0].legend(fontsize=6)

        for ii, xvar in enumerate(com_vars):
            df2 = fnn.loc[(fnn.id==xid) & (fnn.xvar==xvar), :]

            # plot 
            ax[ii, 1].plot(edims, df2.iloc[:,4:].T, alpha=0.75)
            ax[ii, 1].set_xlabel('$\it{m}$')
            ax[ii, 1].set_xticks(np.arange(2, 10, 2))
            ax[ii, 1].set_title(com_vars[xvar], fontsize=8)
            ax[ii, 1].set_title(xid, loc='left', fontsize=6)

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_FIGS + '03_4_bbi__state_space.pdf') as pdf:
    # state space for channel-wise EEG and COM data 
    #   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
    #   - COM data [AP, ML, VT, RES(AP, ML, VT)]

    for xid in EEG:
        for lobe in eeg_channels:
            fig, ax = plt.subplots(len(eeg_bands), len(com_vars),
                                   figsize=(11.5, 7), sharex=True, sharey=True)
            for ii, band in enumerate(eeg_bands):
                for j, xvar in enumerate(com_vars):
                    # com data 
                    tcom = COM[xid]['time']
                    f = (tcom > trange[0]) & (tcom < trange[1])
                    com = bbi.normalize(COM[xid][xvar], scale_max=True)[f]
                    Acom = nld.Amat(com, m_com, tau_com)  # build trajectory matrix

                    for ic, channel in enumerate(eeg_channels[lobe]):
                        # eeg data
                        teeg = EEG[xid]['time'].reshape(-1)
                        f = (teeg > trange[0]) & (teeg < trange[1])
                        eeg = bbi.normalize(EEG[xid][band][:, channel-1], scale_max=True)[f]
                        Aeeg = nld.Amat(eeg, m_eeg, tau_eeg)  # build trajectory matrix
                        
                        # plot 
                        if ic==0: ax[ii, j].plot(Acom[:,0], Acom[:,1], lw=0.5, alpha=0.15, label=com_vars[xvar])
                        ax[ii, j].plot(Aeeg[:,0], Aeeg[:,1], lw=0.5, alpha=0.5, label=eeg_channels[lobe][channel])
                        ax[ii, j].set_xlabel('time (s)')
                        ax[ii, j].set_title(band, fontsize=8)
                        ax[ii, j].set_title(xid, loc='left', fontsize=6)
                        ax[ii, j].set_title(lobe, loc='right', fontsize=6)
                        ax[ii, j].legend(fontsize=6)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


with PdfPages(PATH_FIGS + '03_5_bbi__state_space_agg.pdf') as pdf:
    # state space for aggregated EEG and COM data
    #   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
    #   - COM data [AP, ML, VT, RES(AP, ML, VT)]

    for xid in EEG_AGG:
        fig, ax = plt.subplots(len(eeg_bands), len(com_vars),
                               figsize=(11.5, 7), sharex=True, sharey=True)
        for k, lobe in enumerate(eeg_channels):
            for ii, band in enumerate(eeg_bands):
                for j, xvar in enumerate(com_vars):
                    # com data 
                    tcom = COM[xid]['time']
                    f = (tcom > trange[0]) & (tcom < trange[1])
                    com = bbi.normalize(COM[xid][xvar], scale_max=True)[f]
                    Acom = nld.Amat(com, m_com, tau_com)  # build trajectory matrix

                    # eeg data
                    teeg = EEG_AGG[xid][lobe]['time'].reshape(-1)
                    f = (teeg > trange[0]) & (teeg < trange[1])
                    eeg = bbi.normalize(EEG_AGG[xid][lobe][band], scale_max=True)[f]
                    Aeeg = nld.Amat(eeg, m_eeg, tau_eeg)  # build trajectory matrix

                    # plot
                    if k==0: ax[ii, j].plot(Acom[:,0], Acom[:,1], lw=0.5, alpha=0.15, label=com_vars[xvar])
                    ax[ii, j].plot(Aeeg[:,0], Aeeg[:,1], lw=0.5, alpha=0.5, label=lobe)
                    ax[ii, j].set_xlabel('time (s)')
                    ax[ii, j].set_title(band, fontsize=8)
                    ax[ii, j].set_title(xid, loc='left', fontsize=6)
                    ax[ii, j].legend(fontsize=6)

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_FIGS + '03_6_eeg__state_space.pdf') as pdf:
    # state space for channel-wise EEG and COM data 
    #   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
    #   - COM data [AP, ML, VT, RES(AP, ML, VT)]

    for xid in EEG:
        for lobe in eeg_channels:
            counter = 1
            fig = plt.figure(figsize=(11.5, 7))
            for ii, band in enumerate(eeg_bands):
                    for ic, channel in enumerate(eeg_channels[lobe]):
                        ax = fig.add_subplot(2, 6, counter, projection='3d')
                        # eeg data
                        teeg = EEG[xid]['time'].reshape(-1)
                        f = (teeg > trange[0]) & (teeg < trange[1])
                        eeg = bbi.normalize(EEG[xid][band][:, channel-1], scale_max=True)[f]
                        Aeeg = nld.Amat(eeg, m_eeg, tau_eeg)  # build trajectory matrix
                        
                        s = np.abs((2 * ((Aeeg[:,1]-np.min(Aeeg[:,1]) / (np.max(Aeeg[:,1])-np.min(Aeeg[:,2])))) + 0.25) - 2)
                        c = 220 * ((Aeeg[:,2]-np.min(Aeeg[:,2]) / (np.max(Aeeg[:,2])-np.min(Aeeg[:,2])))) + 10
                        
                        # plot 
                        ax.scatter3D(Aeeg[:,0], Aeeg[:,1], Aeeg[:,2], c=c, s=s,
                                     alpha=0.25, label=eeg_channels[lobe][channel])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_zticks([])
                        ax.set_title(band, fontsize=8)
                        ax.set_title(xid, loc='left', fontsize=6)
                        ax.set_title(lobe, loc='right', fontsize=6)
                        ax.legend(fontsize=6)
                        
                        counter += 1

            plt.tight_layout()
            pdf.savefig()
            plt.close()


# end 
