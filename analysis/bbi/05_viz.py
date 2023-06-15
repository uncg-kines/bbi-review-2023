# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - vizualizations for cross sample entropy 
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import pandas as pd 
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


# define paths & load data
PATH_DATA = './data/nld/'
PATH_FIGS = './output/dat/'
# define file
FILE_XSE_AGG = '05_bbi__xse_agg.xlsx'
FILE_XSE = '05_bbi__xse.xlsx'
FILE_CRQA_AGG = '05_bbi__xse_agg.xlsx'
FILE_CRQA = '05_bbi__xse.xlsx'


# select ids for analysis
IDX = {'H05', 'H07', 'C01'}

# total EEG channels 
N_CHANNELS = 32 

# EEG bands
eeg_bands = ('SCO', 'alpha', 'beta', 'gamma')

# EEG channels (by hemisphere)
eeg_channels = {'left': {8: 'C3', 6: 'FC5', 11: 'CP5'},
                'right': {25: 'C4', 28: 'FC6', 22: 'CP6'}}

# center of mass (COM)
com_vars = {'ap__m_s2': 'AP ($m/s^2$)', 'ml__m_s2': 'ML ($m/s^2$)', 
            'vt__m_s2': 'VT ($m/s^2$)', 'res__m_s2': 'RES ($m/s^2$)'}

se_labels = {'m1': 'm=1', 'm2': 'm=2', 'm3': 'm=3', 
             'm4': 'm=4', 'm5': 'm=5'}
# max embedding
max_m = 5

# import data
xse = pd.read_excel(PATH_DATA + FILE_XSE)
xse_agg = pd.read_excel(PATH_DATA + FILE_XSE_AGG)


with PdfPages(PATH_FIGS + '05_1_bbi__xse.pdf') as pdf:
    # cross-sample entropy plots
    #   - Left: C3, FC5, CP5
    #   - Right: C4, FC6, CP6)
    #   - AP, ML, VT, RES(AP, ML, VT)

    for xid in IDX:
        for lobe in eeg_channels:
            for ic, channel in enumerate(eeg_channels[lobe]):
                fig, ax = plt.subplots(4, 4, figsize=(11.5, 7))
                plt.rcParams.update({'font.size': 8})
                
                for ii, band in enumerate(eeg_bands):
                    for j, xvar in enumerate(com_vars):
                        
                            f = (xse.id==xid) & (xse.band==band) & \
                                (xse.channel_num==channel) & (xse.com_var==xvar)
                            df = xse.loc[f, :]

                            for m in range(2, max_m):
                                m = 'm{}'.format(m)

                                ax[ii, j].plot(df.radius, df.loc[:,m], lw=0.75, label=m)
                                ax[ii, j].set_ylim(0, 5)
                                ax[ii, j].set_xlabel('$r$')
                                ax[ii, j].set_ylabel('cross-SampEn')
                                ax[ii, j].set_title('{}: {}'.format(band, com_vars[xvar]))
                                ax[ii, j].set_title(xid, loc='left', fontsize=6)
                                ax[ii, j].set_title(eeg_channels[lobe][channel], loc='right', fontsize=6)
                                ax[ii, j].legend(fontsize=6)

                plt.tight_layout()
                pdf.savefig()
                plt.close()


with PdfPages(PATH_FIGS + '05_2_bbi__xse_agg.pdf') as pdf:
    # cross-sample entropy plots, aggregated data
    #   - Left: C3, FC5, CP5
    #   - Right: C4, FC6, CP6)
    #   - AP, ML, VT, RES(AP, ML, VT)
    for xid in IDX:
        for lobe in eeg_channels:
            fig, ax = plt.subplots(4, 4, figsize=(11.5, 7))
            plt.rcParams.update({'font.size': 8})

            for ii, band in enumerate(eeg_bands):
                for j, xvar in enumerate(com_vars):
                    f = (xse_agg.id==xid) & (xse_agg.band==band) & \
                        (xse_agg.lobe==lobe) & (xse_agg.com_var==xvar)
                    df = xse_agg.loc[f, :]

                    for m in range(2, max_m+1):
                        m = 'm{}'.format(m)

                        ax[ii, j].plot(df.radius, df.loc[:,m], lw=0.75, label=m)
                        ax[ii, j].set_ylim(0, 3.5)
                        ax[ii, j].set_xlabel('$r$')
                        ax[ii, j].set_ylabel('cross-SampEn')
                        ax[ii, j].set_title('{}: {}'.format(band, com_vars[xvar]))
                        ax[ii, j].set_title(xid, loc='left', fontsize=6)
                        ax[ii, j].set_title(lobe, loc='right', fontsize=6)
                        ax[ii, j].legend(fontsize=6)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


# add CRQA plots ... 


# end 
