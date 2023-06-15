# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script calcualtes the false nearest neighbors for 
#     the EEG and COM data 
#   - plotting is performed in <03_state_space.py> 
#   - check the excel export to ensure it's not commented out
#     if running this yourself
#
# WARNING:
#   - the false nearest neighbors calculations will take a 
#     while to run... grab a coffee :) 
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import numpy as np
import pandas as pd

from src import fun_bbi as bbi
from src import nonlinear_dynamics as nld
from src.params import IDX, eeg_bands, eeg_channels, com_vars, \
    tau_eeg, tau_com, max_m_fnn, trange


# define paths
PATH_DATA = './data/npy/'
PATH_OUT = './data/nld/'
PATH_FIGS = './output/dat/'

# load data... 
data = bbi.import_npz(PATH_DATA)
COM, EEG = data['COM_128'], data['EEG_128']
EEG_AGG = bbi.agg_bands(EEG, eeg_bands, eeg_channels)



# false nearest neighbors for aggregated EEG and COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
nx = len(IDX)*len(eeg_channels)*len(eeg_bands) + len(IDX)*len(com_vars)
demo = ['id', 'lobe', 'band', 'xvar']
edims = ['m{}'.format(ii) for ii in range(1, max_m_fnn+1)]
cols = demo + edims
fnn_out = pd.DataFrame(np.zeros([nx, len(cols)]), columns=cols)

row = 0
for xid in IDX:
    for xvar in com_vars:
        print('\n', xid, xvar)

        # com data 
        tcom = COM[xid]['time']
        f = (tcom > trange[0]) & (tcom < trange[1])
        com = bbi.normalize(COM[xid][xvar])[f]
        Acom = nld.Amat(com, max_m_fnn, tau_com)  # build trajectory matrix

        # calculate fnn
        fnn_com = nld.fnn(Acom, tol=20)

        # index output
        fnn_out.iloc[row, :4] = [xid, np.nan, np.nan, xvar]
        fnn_out.iloc[row, 4:] = fnn_com[1]

        row +=1

    for lobe in eeg_channels:
        for band in eeg_bands:
            print('\n', xid, lobe, band)

            # eeg data
            teeg = EEG_AGG[xid][lobe]['time'].reshape(-1)
            f = (teeg > trange[0]) & (teeg < trange[1])
            eeg = bbi.normalize(EEG_AGG[xid][lobe][band])[f]
            Aeeg = nld.Amat(eeg, max_m_fnn, tau_eeg)  # build trajectory matrix

            # calculate false nearest neighbors 
            fnn_eeg = nld.fnn(Aeeg, tol=20)

            # index output
            fnn_out.iloc[row, :4] = [xid, lobe, band, np.nan]
            fnn_out.iloc[row, 4:] = fnn_eeg[1]

            row +=1

fnn_out
# fnn_out.to_excel(PATH_OUT + '03_fnn.xlsx', index=False)

# end 
