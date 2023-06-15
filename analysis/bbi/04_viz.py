# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - produce vizualizaitons for sample entropy
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from src.params import IDX, eeg_bands, eeg_channels, com_vars, \
    se_labels, max_m_se, m_com, m_eeg, r_se

# define paths
PATH_DATA = './data/nld/'
PATH_FIGS = './output/dat/'
# define data files 
FILE_EEG_SE = '04_eeg__se_opt.xlsx'
FILE_EEG_SE_AGG = '04_eeg__se_opt_agg.xlsx'
FILE_COM_SE = '04_com__se_opt.xlsx'


# load data...
eeg_se = pd.read_excel(PATH_DATA + FILE_EEG_SE)
eeg_se_agg = pd.read_excel(PATH_DATA + FILE_EEG_SE_AGG)
com_se = pd.read_excel(PATH_DATA + FILE_COM_SE)


# -------------------------------------------------------
with PdfPages(PATH_FIGS + '04_1_eeg__se_opt.pdf') as pdf:
    # plot sample entropy for chanel-wise EEG data 
    #   - Left: C3, FC5, CP5
    #   - Right: C4, FC6, CP6)

    for xid in IDX:
        for lobe in eeg_channels:
            fig = plt.figure(figsize=(11, 8))
            plt.rcParams.update({'font.size': 8})

            counter = 1
            for ii, band in enumerate(eeg_bands):
                for channel in eeg_channels[lobe]:
                    f = (eeg_se.id==xid) & (eeg_se.band==band) & (eeg_se.channel_num==channel)
                    df = eeg_se.loc[f, :]

                    ax = plt.subplot(4, 3, counter)
                    
                    for m in range(2, max_m_se+1):
                        m = 'm{}'.format(m)

                        ax.plot(df.radius, df.loc[:,m], lw=0.75, label=se_labels[m])
                        ax.set_ylim(0, 2)
                        ax.set_xlabel('$r$')
                        ax.set_ylabel('sample entropy')
                        ax.set_title('{}'.format(band))
                        ax.set_title(xid, loc='left', fontsize=6)
                        ax.set_title('ch: {}'.format(channel), loc='right', fontsize=6)
                        ax.legend(fontsize=6)
                        
                    counter += 1

            plt.tight_layout()
            pdf.savefig()
            plt.close()


with PdfPages(PATH_FIGS + '04_2_eeg__se_opt_agg.pdf') as pdf:
    # plot sample entropy for aggregated EEG data 
    #   - Left: C3, FC5, CP5
    #   - Right: C4, FC6, CP6)

    for xid in IDX:
        fig, ax = plt.subplots(4, 2, figsize=(5, 8))
        plt.rcParams.update({'font.size': 8})

        for ii, band in enumerate(eeg_bands):
            for j, lobe in enumerate(('left', 'right')): # separate left and right 
                f = (eeg_se_agg.id==xid) & (eeg_se_agg.lobe==lobe) & (eeg_se_agg.band==band) 
                df = eeg_se_agg.loc[f, :]
                
                for m in range(2, max_m_se+1):
                    m = 'm{}'.format(m)

                    ax[ii, j].plot(df.radius, df.loc[:, m], lw=0.75, label=se_labels[m])
                    ax[ii, j].set_ylim(0, 2)
                    ax[ii, j].set_xlabel('$r$')
                    ax[ii, j].set_ylabel('sample entropy')
                    ax[ii, j].set_title('{}'.format(band))
                    ax[ii, j].set_title(xid, loc='left', fontsize=6)
                    ax[ii, j].set_title(lobe, loc='right', fontsize=6)
                    ax[ii, j].legend(fontsize=6)

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_FIGS + '04_3_com__se_opt.pdf') as pdf:
    # sample entropy for COM data
    #   - AP, ML, VT, RES(AP, ML, VT)

    for xid in IDX:
        fig = plt.figure(figsize=(3.5, 8))
        plt.rcParams.update({'font.size': 8})
            
        for j, xvar in enumerate(com_vars):
            # flag and subset data
            f = (com_se.id==xid) & (com_se.com_var==xvar)
            df = com_se.loc[f, :]
            
            for m in range(2, max_m_se+1):
                m = 'm{}'.format(m)
                
                ax = plt.subplot(4, 1, j+1)
                ax.plot(df.radius, df.loc[:, m], '-', lw=0.75, label=se_labels[m])
                ax.set_ylim(0, 3.)
                ax.set_xlabel('$r$')
                ax.set_ylabel('sample entropy')
                ax.set_title(com_vars[xvar])
                ax.set_title(xid, loc='left', fontsize=6)
                ax.legend(fontsize=6)

        plt.tight_layout()
        pdf.savefig()
        plt.close()


with PdfPages(PATH_FIGS + '04_4_eeg__se_hm.pdf') as pdf:
    # plot sample entropy for chanel-wise EEG data 
    #   - Left: C3, FC5, CP5
    #   - Right: C4, FC6, CP6)
    
    vmin, vmax = 0, 2.25     # min/max for heatmap
    fig, ax = plt.subplots(3, 3, figsize=(9, 5.6))
    plt.rcParams.update({'font.size': 8})
    for k, xid in enumerate(IDX):

        se_mat = np.zeros([4, 3])
        for jj, lobe in enumerate(eeg_channels):
            for j, channel in enumerate(eeg_channels[lobe]):
                for ii, band in enumerate(eeg_bands):
                    f = (eeg_se.id==xid) & (eeg_se.band==band) & \
                        (eeg_se.channel_num==channel) & (eeg_se.radius==0.20)
                    se_mat[ii, j] = eeg_se.loc[f, 'm{}'.format(m_eeg)].to_numpy()

            ax[k, jj].imshow(se_mat, vmin=vmin, vmax=vmax, cmap='binary')
            xticks, yticks = np.arange(0, 3), np.arange(0, 4)
            ax[k, jj].set_xticks(xticks, labels=eeg_channels[lobe].values())
            ax[k, jj].set_yticks(yticks, labels=eeg_bands)
            ax[k, jj].set_title(xid, loc='left', fontsize=6)
        
        se_mat = np.zeros([4, 1])
        for ii, xvar in enumerate(com_vars):
            f = (com_se.id==xid) & (com_se.com_var==xvar) & (com_se.radius==r_se)
            se_mat[ii, 0] = com_se.loc[f, 'm{}'.format(m_eeg)].to_numpy()

        ax[k, 2].imshow(se_mat, vmin=vmin, vmax=vmax, cmap='binary')
        ax[k, 2].set_xticks([])
        ax[k, 2].set_yticks(yticks, labels=com_vars.values())
        ax[k, 2].set_title(xid, loc='left', fontsize=6)

    plt.tight_layout()
    pdf.savefig()
    plt.close()


with PdfPages(PATH_FIGS + '04_4_eeg__se_hm_agg.pdf') as pdf:
    # plot sample entropy for chanel-wise EEG data 
    #   - Left: C3, FC5, CP5
    #   - Right: C4, FC6, CP6)
    
    vmin, vmax = 0., 2.25     # min/max for heatmap
    fig, ax = plt.subplots(3, 2, figsize=(5, 5.6))
    plt.rcParams.update({'font.size': 8})
    for k, xid in enumerate(IDX):

        se_mat = np.zeros([4, 2])
        for jj, lobe in enumerate(eeg_channels):
            for ii, band in enumerate(eeg_bands):
                f = (eeg_se_agg.id==xid) & (eeg_se_agg.lobe==lobe) & \
                    (eeg_se_agg.band==band) & (eeg_se_agg.radius==0.20)
                se_mat[ii, jj] = eeg_se_agg.loc[f, 'm{}'.format(m_com)].to_numpy()

        ax[k, 0].imshow(se_mat, vmin=vmin, vmax=vmax, cmap='binary')
        xticks, yticks = np.arange(0, 2), np.arange(0, 4)
        ax[k, 0].set_xticks(xticks, labels=eeg_channels)
        ax[k, 0].set_yticks(yticks, labels=eeg_bands)
        ax[k, 0].set_title(xid, loc='left', fontsize=6)
        
        se_mat = np.zeros([4, 1])
        for ii, xvar in enumerate(com_vars):
            f = (com_se.id==xid) & (com_se.com_var==xvar) & (com_se.radius==r_se)
            se_mat[ii, 0] = com_se.loc[f, 'm{}'.format(m_com)].to_numpy()

        ax[k, 1].imshow(se_mat, vmin=vmin, vmax=vmax, cmap='binary')
        ax[k, 1].set_xticks([])
        ax[k, 1].set_yticks(yticks, labels=com_vars.values())
        ax[k, 1].set_title(xid, loc='left', fontsize=6)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

# end 
