# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script calcualtes sample entropy for EEG and COM data 
#   - calulations are made across a variety of embeddings and radii 
#     with a fixed lag (tau)s
#
# WARNING: 
#   - runtime on these scripts is not insignificant... 
#     Grab a coffee or go for a walk :) 
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import numpy as np
import pandas as pd 
import EntropyHub as EH

from src import fun_bbi as bbi
from src.params import IDX, eeg_bands, eeg_channels, com_vars, \
    max_m_se, tau_eeg, tau_com, radii

# define paths
PATH_DATA = './data/npy/'
PATH_OUT = './data/'

# load data...
data = bbi.import_npz(PATH_DATA, verbose=False)
COM, EEG = data['COM_128'], data['EEG_128']
EEG_AGG = bbi.agg_bands(EEG, eeg_bands, eeg_channels) # agg data 



# EEG ------------------------------------------------------
# initiate dataframe for analysis of channel-wise EEG 
nx = len(IDX)*len(eeg_bands)*6*len(radii)
cols = ['id', 'band', 'channel_num', 'radius', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5']
eeg_se_opt = pd.DataFrame(np.zeros([nx, len(cols)]), columns=cols)

row = 0
for xid in IDX:
    for band in eeg_bands:
        for lobe in eeg_channels:
            for channel in eeg_channels[lobe]:
                print(xid, band, channel)

                x = EEG[xid]['time']
                y = bbi.normalize(EEG[xid][band][:, channel-1])

                for ii_r, r in enumerate(radii):
                    se = EH.SampEn(y, m=max_m_se, r=r, tau=tau_eeg)[0]

                    eeg_se_opt.loc[row, cols[:4]] = [xid, band, channel, r]
                    eeg_se_opt.loc[row, cols[4:]] = se

                    row += 1

eeg_se_opt
# eeg_se_opt.to_excel(PATH_OUT + 'nld/04_eeg__se_opt.xlsx', index=False)


# initiate dataframe for analysis of aggregated bands... 
nx = len(IDX)*len(eeg_bands)*len(eeg_channels)*len(radii)
cols = ['id', 'lobe', 'band', 'radius', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5']
eeg_se_opt2 = pd.DataFrame(np.zeros([nx, len(cols)]), columns=cols)

row = 0
for xid in IDX:
    for k, lobe in enumerate(eeg_channels):
        for band in eeg_bands:
            print(xid, band, lobe)
            if k == 1: k=5

            x = EEG_AGG[xid][lobe]['time']
            y = bbi.normalize(EEG_AGG[xid][lobe][band])

            for ii_r, r in enumerate(radii):
                se = EH.SampEn(y, m=max_m_se, r=r, tau=tau_com)[0]

                eeg_se_opt2.loc[row, cols[:4]] = [xid, lobe, band, r]
                eeg_se_opt2.loc[row, cols[4:]] = se

                row += 1

eeg_se_opt2
# eeg_se_opt2.to_excel(PATH_OUT + 'nld/04_eeg__se_opt_agg.xlsx', index=False)


# COM ------------------------------------------------------
# initiate dataframe for analysis of COM data 
nx = len(IDX)*len(com_vars)*len(radii)
cols = ['id', 'com_var', 'radius', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5']
com_se_opt = pd.DataFrame(np.zeros([nx, len(cols)]), columns=cols)

row = 0
for ii, xid in enumerate(IDX):
    for k, xvar in enumerate(com_vars.keys()):
        print(xid, xvar)

        x = COM[xid]['time']
        y = bbi.normalize(COM[xid][xvar])

        # loop sample entropy
        for ii_r, r in enumerate(radii):
            se = EH.SampEn(y, max_m_se, r=r, tau=tau_com)[0]

            com_se_opt.loc[row, cols[:3]] = [xid, xvar, r]
            com_se_opt.loc[row, cols[3:]] = se

            row += 1

com_se_opt
# com_se_opt.to_excel(PATH_OUT + 'nld/04_com__se_opt.xlsx', index=False)

# end
