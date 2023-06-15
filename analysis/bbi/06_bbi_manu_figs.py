# nate berry
# 2023-01-11
#
# -----------------------------------
# NOTES:
# - plots for manuscript 
# 
# - somatomotor channels
#       - RIGHT [25, 22, 28]
#       - LEFT [8, 6, 11]
#
# -----------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Cross
from pyrqa.settings import Settings
from pyrqa.computation import RPComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius

from src import fun_bbi as bbi
from src import nonlinear_dynamics as nld
from src.params import *


# define paths & load data --------------------
PATH_DATA_NPY = './data/npy/'
PATH_DATA_NLD = './data/nld/'
PATH_FIGS = './output/manu/'

# file names of NLD data 
# fnn data 
FILE_FNN = '03_fnn.xlsx'
# univariate time series
FILE_COM_SE = '04_com__se_opt.xlsx'
FILE_COM_RQA = '04_com__rqa.xlsx'
FILE_EEG_SE_AGG = '04_eeg__se_opt_agg.xlsx'
FILE_EEG_SE = '04_com__se_opt.xlsx'
FILE_COM_RQA_AGG = '04_com__rqa_agg.xlsx'
FILE_EEG_RQA = '04_eeg__rqa.xlsx'
# crossed time series 
FILE_XSE_AGG = '05_bbi__xse_agg.xlsx'
FILE_XSE = '05_bbi__xse.xlsx'
FILE_CRQA_AGG = '05_bbi__crqa_agg.xlsx'
FILE_CRQA = '05_bbi__crqa.xlsx'


# import raw data 
data = bbi.import_npz(PATH_DATA_NPY)
COM, EEG = data['COM_128'], data['EEG_128']
EEG_AGG = bbi.agg_bands(EEG, eeg_bands, eeg_channels)

# import fnn data
fnn = pd.read_excel(PATH_DATA_NLD + FILE_FNN)
# import NLD data (univiariate time series)
com_se = pd.read_excel(PATH_DATA_NLD + FILE_COM_SE)
com_rqa = pd.read_excel(PATH_DATA_NLD + FILE_COM_RQA)
eeg_se_agg = pd.read_excel(PATH_DATA_NLD + FILE_EEG_SE_AGG)
eeg_se = pd.read_excel(PATH_DATA_NLD + FILE_EEG_SE)
eeg_rqa = pd.read_excel(PATH_DATA_NLD + FILE_EEG_RQA)
# import NLD data (crossed time series)
xse_agg = pd.read_excel(PATH_DATA_NLD + FILE_XSE_AGG)
crqa_agg = pd.read_excel(PATH_DATA_NLD + FILE_CRQA_AGG)



# ------------------------------------------------------------
# plot the raw data EEG & COM data
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#       * channel-wise and aggregate 
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
for xid in IDX:
    fig, ax = plt.subplots(4, 3, figsize=(8, 5.6), sharex=True)
    plt.rcParams.update({'font.size':8})

    for j, lobe in enumerate(eeg_channels):
        for ii, band in enumerate(eeg_bands):
            for channel in eeg_channels[lobe]:
                t = EEG[xid]['time']
                eeg = EEG[xid][band][:, channel-1]
                eeg_agg = EEG_AGG[xid][lobe][band]
                label = eeg_channels[lobe][channel]

                ax[ii, j].plot(t, eeg, lw=0.3, alpha=0.75, label=label)
                ax[ii, j].plot(t, eeg_agg, 'k--', lw=0.35, alpha=0.8)
                if ii == 3: ax[ii, j].set_xlabel('time (s)')
                ax[ii, j].set_ylabel('power ($\mu V^2/Hz$)')
                ax[ii, j].set_title(band)
                ax[ii, j].set_title(xid, loc='left', fontsize=7)
                ax[ii, j].set_title(lobe, loc='right', fontsize=7)
                ax[ii, j].legend(fontsize=5)

    for ii, xvar in enumerate(com_vars):
        t = COM[xid]['time']
        com = COM[xid][xvar]
        label = com_vars[xvar]

        ax[ii, 2].plot(t, com, lw=0.3)
        if ii == 3: ax[ii, 2].set_xlabel('time (s)')
        ax[ii, 2].set_ylabel('Accel. $(m/s^2)$')
        ax[ii, 2].set_title(com_vars[xvar])
        ax[ii, 2].set_title(xid, loc='left', fontsize=7)

    plt.tight_layout()
    plt.savefig(PATH_FIGS + '01_dat_{}.jpg'.format(xid), dpi=500)
    # plt.show()


# ------------------------------------------------------------
# plot acf and ami for the EEG & COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#       * channel-wise and aggregate 
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
for xid in IDX:
    fig, ax = plt.subplots(4, 4, figsize=(8, 5.6))
    plt.rcParams.update({'font.size':8})

    for j, lobe in enumerate(eeg_channels):
        for ii, band in enumerate(eeg_bands):
            counter = 0
            for channel in eeg_channels[lobe]:
                print('\n', xid, lobe, band, channel, '\n')

                # subset time and data 
                t = EEG[xid]['time']
                eeg = EEG[xid][band][:, channel-1]
                eeg_agg = EEG_AGG[xid][lobe][band]

                # calclulate acf and ami 
                acorr = nld.acf(eeg, maxlags=mlag_acf)
                if counter==2: acorr2 = nld.acf(eeg_agg, maxlags=mlag_acf)
                mi = nld.ami(eeg, maxlags=mlag_ami)
                if counter==2: mi2 = nld.ami(eeg_agg, maxlags=mlag_ami)

                # plot ACF
                label = eeg_channels[lobe][channel]
                ax[ii, j].plot(acorr[0], acorr[1], lw=0.6, alpha=0.6, label=label)
                ax[ii, j].fill_between(acorr[0], acorr[1], alpha=0.1)
                if counter==2: ax[ii, j].plot(acorr2[0], acorr2[1], 'k', lw=0.75)
                if ii == 3: ax[ii, j].set_xlabel('lags')
                ax[ii, j].set_ylabel('ACF')
                ax[ii, j].set_title(band)
                ax[ii, j].set_title(xid, loc='left', fontsize=7)
                ax[ii, j].set_title(lobe, loc='right', fontsize=7)
                ax[ii, j].legend(fontsize=5)

                # plot AMI
                ax[ii, j+2].plot(mi[0], mi[1], lw=0.6, alpha=0.6, label=label)
                ax[ii, j+2].fill_between(mi[0], mi[1], alpha=0.1)
                if counter==2: ax[ii, j+2].plot(mi2[0], mi2[1], 'k', lw=0.75)
                if ii == 3: ax[ii, j+2].set_xlabel('lags')
                ax[ii, j+2].set_ylabel('AMI')
                ax[ii, j+2].set_title(band)
                ax[ii, j+2].set_title(xid, loc='left', fontsize=7)
                ax[ii, j+2].set_title(lobe, loc='right', fontsize=7)
                ax[ii, j+2].legend(fontsize=5)

                counter += 1

    plt.tight_layout()
    plt.savefig(PATH_FIGS + '02_acf_ami_{}.jpg'.format(xid), dpi=500)
    # plt.show()
 

# ------------------------------------------------------------
# false nearest neighbors for aggregate EEG & COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#       * channel-wise and aggregate 
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
edims = np.arange(1, 11)
fig, ax = plt.subplots(len(eeg_bands), len(IDX)*2,
                       figsize=(8, 5.6), sharex=True, sharey=True)

for j, xid in enumerate(IDX):
    j = j*2

    for ii, band in enumerate(eeg_bands):
        for k, lobe in enumerate(eeg_channels):
            ls = ('-', ':')[k]
            df = fnn.loc[(fnn.id==xid) & (fnn.lobe==lobe) & (fnn.band==band), :]

            # plot 
            ax[ii, j].plot(edims, df.iloc[:,4:].T, lw=1, alpha=0.75, label=lobe)
            if ii==3: ax[ii, j].set_xlabel('$\it{m}$')
            ax[ii, j].set_xticks(np.arange(2, 10, 2))
            ax[ii, j].set_title(band, fontsize=8)
            ax[ii, j].set_title(xid, loc='left', fontsize=6)
            ax[ii, j].legend(fontsize=6)

    for ii, xvar in enumerate(com_vars):
        df2 = fnn.loc[(fnn.id==xid) & (fnn.xvar==xvar), :]

        # plot 
        ax[ii, j+1].plot(edims, df2.iloc[:,4:].T, lw=0.75, alpha=0.5)
        if ii==3: ax[ii, j+1].set_xlabel('$\it{m}$')
        ax[ii, j+1].set_xticks(np.arange(2, 10, 2))
        ax[ii, j+1].set_title(com_vars[xvar], fontsize=8)
        ax[ii, j+1].set_title(xid, loc='left', fontsize=6)

plt.tight_layout()
plt.savefig(PATH_FIGS + '03_fnn.jpg', dpi=500)


# ------------------------------------------------------------
# state space reconstruction for aggregate EEG & COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
fig, ax = plt.subplots(4, 6, figsize=(8, 5.6)) # sharex=True, sharey=True)
plt.rcParams.update({'font.size':8})
j = 0
for xid in IDX:
    for lobe in eeg_channels:
        for ii, band in enumerate(eeg_bands):
            print('\n', xid, lobe, band, '\n')

            fc, order = 6, 4

            # eeg signal 
            f = (EEG_AGG[xid][lobe]['time'] > trange[0]) & (EEG_AGG[xid][lobe]['time'] < 120)
            eeg = bbi.normalize(EEG_AGG[xid][lobe][band], scale_max=True)[f]
            eeg = bbi.lowpass_butter(eeg, fc, fs, order)

            # build trajectories and calculate fnn 
            A = nld.Amat(eeg, m_eeg, tau_eeg)

            # plot 
            ax[ii, j].plot(A[:, 0], A[:, -1], lw=0.2, alpha=0.75, label=lobe)
            if ii==3: ax[ii, j].set_xlabel('x(i)')
            if j==0: ax[ii, j].set_ylabel(r'x(i+2$\tau}$)')
            ax[ii, j].set_xticks([])
            ax[ii, j].set_yticks([])
            ax[ii, j].set_title(band)
            ax[ii, j].set_title(xid, loc='left', fontsize=7)
            ax[ii, j].legend(fontsize=5)

    for ii, xvar in enumerate(com_vars):
        # com data
        f = (COM[xid]['time'] > trange[0]) & (COM[xid]['time'] < trange[1])
        com = bbi.normalize(COM[xid][xvar], scale_max=True)[f]
        com = bbi.lowpass_butter(com, fc, fs, order)

        # build trajectories and calculate fnn
        A = nld.Amat(com, m_com, tau_com)
 
        # plot 
        ax[ii, j+1].plot(A[:, 0], A[:, -1], lw=0.2, alpha=0.75)
        if ii==3: ax[ii, j+1].set_xlabel('x(i)')
        ax[ii, j+1].set_xticks([])
        ax[ii, j+1].set_yticks([])
        ax[ii, j+1].set_title(com_vars[xvar])
        ax[ii, j+1].set_title(xid, loc='left', fontsize=7)
    
    j += 2

plt.tight_layout()
plt.savefig(PATH_FIGS + '04_state_space.jpg', dpi=500)
# plt.show()


# ------------------------------------------------------------
# plot sample entropy for aggregated EEG data 
#   - Left: C3, FC5, CP5
#   - Right: C4, FC6, CP6)
fig, ax = plt.subplots(4, 6, figsize=(8, 5.6), sharex=True, sharey=True)
plt.rcParams.update({'font.size': 8})
j = 0
for xid in IDX:
    for lobe in eeg_channels:
        for ii, band in enumerate(eeg_bands):
            f = (eeg_se_agg.id==xid) & (eeg_se_agg.lobe==lobe) & (eeg_se_agg.band==band) 
            df = eeg_se_agg.loc[f, :]
            
            for m in range(2, max_m_se+1):
                m = 'm{}'.format(m)

                ax[ii, j].plot(df.radius, df.loc[:, m], lw=0.75, 
                               label=se_labels[m], alpha=0.75)
                ax[ii, j].set_ylim(0, 2)
                if ii==3: ax[ii, j].set_xlabel('$r$')
                if j==0: ax[ii, j].set_ylabel('sample entropy')
                ax[ii, j].set_xticks(np.arange(0.1, 0.4, 0.1))
                ax[ii, j].set_title('{}'.format(band), fontsize=8)
                ax[ii, j].set_title(xid, loc='left', fontsize=6)
                ax[ii, j].set_title(lobe, loc='right', fontsize=6)
                ax[ii, j].legend(fontsize=5)

        j += 1

plt.tight_layout()
plt.savefig(PATH_FIGS + '05_eeg_se.jpg', dpi=500)
plt.close()


# ------------------------------------------------------------
# sample entropy for COM data
#   - AP, ML, VT, RES(AP, ML, VT)
fig, ax = plt.subplots(4, 3, figsize=(4, 5.6), sharex=True, sharey=True)
plt.rcParams.update({'font.size': 8})
j = 0
for xid in IDX:
    for ii, xvar in enumerate(com_vars):
        # flag and subset data
        f = (com_se.id==xid) & (com_se.com_var==xvar)
        df = com_se.loc[f, :]
        
        for m in range(2, max_m_se+1):
            m = 'm{}'.format(m)
            
            ax[ii, j].plot(df.radius, df.loc[:, m], '-', lw=0.75, label=se_labels[m], alpha=0.75)
            ax[ii, j].set_ylim(0, 3.)
            if ii==3: ax[ii, j].set_xlabel('$r$')
            if j==0: ax[ii, j].set_ylabel('sample entropy')
            ax[ii, j].set_title(com_vars[xvar], fontsize=8)
            ax[ii, j].set_title(xid, loc='left', fontsize=6)
            ax[ii, j].legend(fontsize=5)

    j += 1

plt.tight_layout()
plt.savefig(PATH_FIGS + '06_com_se.jpg', dpi=500)
plt.close()


# ------------------------------------------------------------
# sample entropy and rqa heatmaps (aggregated bands)
#   - Left: C3, FC5, CP5
#   - Right: C4, FC6, CP6)
fig, ax = plt.subplots(3, 4, figsize=(8, 5.6))
plt.rcParams.update({'font.size': 8})
for k, xid in enumerate(IDX):
    vmin, vmax = 0, 1.25
    se_mat = np.zeros([4, 2])
    rp_mat = np.zeros([4, 2])

    for jj, lobe in enumerate(eeg_channels):
        for ii, band in enumerate(eeg_bands):
            mx = eeg_se_agg.loc[:, 'm3'].max()
            f = (eeg_se_agg.id==xid) & (eeg_se_agg.lobe==lobe) & \
                (eeg_se_agg.band==band) & (eeg_se_agg.radius==0.20)
            se_mat[ii, jj] = eeg_se_agg.loc[f, 'm3'].to_numpy() / mx
            
            mx = eeg_rqa.loc[:, 'rec'].max()
            f = (eeg_rqa.id==xid) & (eeg_rqa.lobe==lobe) &  (eeg_rqa.band==band)
            rp_mat[ii, jj] = eeg_rqa.loc[f, 'rec'].to_numpy() / mx

    ax[k, 0].imshow(se_mat, vmin=vmin, vmax=vmax, cmap='binary')
    xticks, yticks = np.arange(0, 2), np.arange(0, 4)
    ax[k, 0].set_xticks(xticks, labels=eeg_channels)
    ax[k, 0].set_yticks(yticks, labels=eeg_bands)
    ax[k, 0].set_title('{}, SampEn'.format(xid), loc='left', fontsize=6)
    
    ax[k, 2].imshow(rp_mat, vmin=0, vmax=vmax, cmap='binary')
    xticks, yticks = np.arange(0, 2), np.arange(0, 4)
    ax[k, 2].set_xticks(xticks, labels=eeg_channels)
    ax[k, 2].set_yticks(yticks, labels=eeg_bands)
    ax[k, 2].set_title('{}, RQA (rec)'.format(xid), loc='left', fontsize=6)

    se_mat = np.zeros([4, 1])
    rp_mat = np.zeros([4, 1])
    for ii, xvar in enumerate(com_vars):
        f = (com_se.id==xid) & (com_se.com_var==xvar) & (com_se.radius==r_se)
        mx = com_se.loc[:, 'm3'].max()
        se_mat[ii, 0] = com_se.loc[f, 'm3'].to_numpy() / mx

        mx = com_rqa.loc[:, 'rec'].max()
        f = (com_rqa.id==xid) & (com_rqa.com_var==xvar)
        rp_mat[ii, 0] = com_rqa.loc[f, 'rec'].to_numpy() / mx

    ax[k, 1].imshow(se_mat, vmin=vmin, vmax=vmax, cmap='binary')
    ax[k, 1].set_xticks([])
    ax[k, 1].set_yticks(yticks, labels=com_vars.values())
    ax[k, 1].set_title('{}, SampEn'.format(xid), loc='left', fontsize=6)

    ax[k, 3].imshow(rp_mat, vmin=vmin, vmax=vmax, cmap='binary')
    ax[k, 3].set_xticks([])
    ax[k, 3].set_yticks(yticks, labels=com_vars.values())
    ax[k, 3].set_title('{}, RQA (rec)'.format(xid), loc='left', fontsize=6)

plt.tight_layout()
plt.savefig(PATH_FIGS + '07_eeg_se_hm.jpg',  dpi=500)
plt.close()


# ------------------------------------------------------------
# cross-sample entropy plots
#   - Left: C3, FC5, CP5
#   - Right: C4, FC6, CP6)
#   - AP, ML, VT, RES(AP, ML, VT)
for xid in IDX:
    fig, ax = plt.subplots(4, 4, figsize=(6, 5.6), sharex=True, sharey=True)
    plt.rcParams.update({'font.size': 8})
    for k, lobe in enumerate(eeg_channels):
        for ii, band in enumerate(eeg_bands):
            for j, xvar in enumerate(com_vars):
                f = (xse_agg.id==xid) & (xse_agg.band==band) & \
                    (xse_agg.lobe==lobe) & (xse_agg.com_var==xvar)
                df = xse_agg.loc[f, :]

                for m in range(2, max_m_se+1):
                    ls = ('-', '--')[k]
                    c = ('C0', 'C1', 'C2', 'C3')[m-2]
                    m = 'm{}'.format(m)

                    if k==0: ax[ii, j].plot(df.radius, df.loc[:,m], c=c, ls=ls, lw=0.75, label=se_labels[m])
                    if k==1: ax[ii, j].plot(df.radius, df.loc[:,m], c=c, ls=ls, lw=0.75)
                    # ax[ii, j].set_ylim(0, 8)
                    ax[ii, j].set_xticks(np.arange(0.1, 0.4, 0.1))
                    if ii==3: ax[ii, j].set_xlabel('$r$')
                    if j==0: ax[ii, j].set_ylabel('cross-SampEn')
                    ax[ii, j].set_title('{}: {}'.format(band, com_vars[xvar]))
                    ax[ii, j].set_title(xid, loc='left', fontsize=6)
                    ax[ii, j].legend(fontsize=5)

    plt.tight_layout()
    plt.savefig(PATH_FIGS + '08_xse_{}.jpg'.format(xid),  dpi=500)
    plt.close()


# ------------------------------------------------------------
# cross-sample entropy heatmaps
#   - Left: C3, FC5, CP5
#   - Right: C4, FC6, CP6)
#   - AP, ML, VT, RES(AP, ML, VT)
fig, ax = plt.subplots(3, 2, figsize=(4.6, 5.6), sharex=True, sharey=True)
plt.rcParams.update({'font.size': 8})
for k, xid in enumerate(IDX):
    vmin, vmax = 0, 1.2  # min/max for heatmap
    se_mat = np.zeros([4, 4])

    for jj, lobe in enumerate(eeg_channels):
        for ii, band in enumerate(eeg_bands):
            for j, xvar in enumerate(com_vars):
                mx = xse_agg.loc[:, 'm3'].max()
                f = (xse_agg.id==xid) & (xse_agg.lobe==lobe) & (xse_agg.band==band) & \
                    (xse_agg.com_var==xvar) & (xse_agg.radius==0.2)
                se_mat[ii, j] = xse_agg.loc[f, 'm3'] / mx

        ax[k, jj].imshow(se_mat, vmin=vmin, vmax=vmax, cmap='binary')
        xytick = np.arange(0, 4)
        ax[k, jj].set_xticks(xytick, labels=com_vars.values())
        ax[k, jj].set_yticks(xytick, labels=eeg_bands)
        ax[k, jj].set_title('{}, {}'.format(xid, lobe), loc='left', fontsize=6)

plt.tight_layout()
plt.savefig(PATH_FIGS + '09_xse_hm.jpg'.format(xid),  dpi=500)
plt.close()


# ------------------------------------------------------------
# cross-recurrence quantificaiton of aggregated bands and COM data
#   - Left: C3, FC5, CP5
#   - Right: C4, FC6, CP6)
#   - AP, ML, VT, RES(AP, ML, VT)
for xid in IDX:
    fig, ax = plt.subplots(4, 8, figsize=(8, 5))
    plt.rcParams.update({'font.size': 8})
    for k, lobe in enumerate(eeg_channels):
        if k==1: k = 4

        for ii, band in enumerate(eeg_bands):
            for j, xvar in enumerate(com_vars):
                print('\n', xid, band, lobe, xvar, '\n')

                # align COM and EEG data 
                channel = ''
                tcom, com, teeg, eeg = bbi.align_data(
                    COM, EEG_AGG, xid, band, lobe, channel, xvar)
                
                # normalize COM and EEG data 
                com = bbi.normalize(com, scale_max=True)
                eeg = bbi.normalize(eeg, scale_max=True)

                # crqa
                y0 = TimeSeries(com, embedding_dimension=m_bbi, time_delay=tau_bbi)
                y1 = TimeSeries(eeg, embedding_dimension=m_bbi, time_delay=tau_bbi)
                ts = (y0, y1)  # definte crossed time series
                settings = Settings(ts,
                                    analysis_type=Cross,
                                    neighbourhood=FixedRadius(r_rqa),
                                    similarity_measure=EuclideanMetric,
                                    theiler_corrector=theiler)

                # recurrence pplot 
                rp_comp = RPComputation.create(settings, verbose=False)
                rp = rp_comp.run().recurrence_matrix_reverse

                ax[ii, j+k].imshow(rp, cmap='binary')
                ax[ii, j+k].set_xticks([])
                ax[ii, j+k].set_yticks([])
                if (j==0) & (k==0): ax[ii, j+k].set_ylabel(eeg_bands[ii])
                if ii==3: ax[ii, j+k].set_xlabel(com_vars[xvar])
                # ax[ii, j+k].set_title('{}: {}'.format(band, com_vars[xvar]), fontsize=8)
                ax[ii, j+k].set_title('{}, {}'.format(xid, lobe), loc='left', fontsize=6)
        
    plt.tight_layout()
    plt.savefig(PATH_FIGS + '10_crqa_agg_{}.jpg'.format(xid),  dpi=500)
    plt.close()


# ------------------------------------------------------------
# cross-recurrence heatmaps
#   - Left: C3, FC5, CP5
#   - Right: C4, FC6, CP6)
#   - AP, ML, VT, RES(AP, ML, VT)
fig, ax = plt.subplots(3, 4, figsize=(8, 5.6), sharex=True, sharey=True)
plt.rcParams.update({'font.size': 8})

crqa_agg = crqa_agg.loc[crqa_agg.id.str.contains('H05|H07|C01'), :]
for q, rqvar in enumerate(('det', 'rec')):
    if q==0: vmin, vmax = 0.8, 1.1    # min/max for heatmap
    if q==1: vmin, vmax = 0., 1.25    # min/max for heatmap
    q = q * 2
    
    for k, xid in enumerate(IDX):
        rq_mat = np.zeros([4, 4])

        for jj, lobe in enumerate(eeg_channels):
            for ii, band in enumerate(eeg_bands):
                for j, xvar in enumerate(com_vars):
                    mx = crqa_agg.loc[:, rqvar].max()
                    f = (crqa_agg.id==xid) & (crqa_agg.lobe==lobe) & \
                        (crqa_agg.band==band) & (crqa_agg.com_var==xvar)
                    rq_mat[ii, j] = crqa_agg.loc[f, rqvar] / mx
                    rq_mat

            rq_mat
            ax[k, jj+q].imshow(rq_mat, vmin=vmin, vmax=vmax, cmap='binary')
            xytick = np.arange(0, 4)
            ax[k, jj+q].set_xticks(xytick, labels=com_vars.values())
            ax[k, jj+q].set_yticks(xytick, labels=eeg_bands)
            ax[k, jj+q].set_title('{}, {}'.format(xid, lobe), loc='left', fontsize=6)
            ax[k, jj+q].set_title(rqvar, loc='right', fontsize=6)

plt.tight_layout()
plt.savefig(PATH_FIGS + '11_crqa_hm.jpg',  dpi=500)
plt.close()

# end 
