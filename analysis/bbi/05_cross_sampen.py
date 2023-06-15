# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script calcualtes cross-sample entropy (cross-SampEn) 
#     for EEG & COM data
#   - calulations are made across a variety of embeddings and radii 
#     with a fixed lag (tau)
#
# WARNING: 
#   - runtime on these scripts is not insignificant... grab a coffee
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import pandas as pd 
import numpy as np
import EntropyHub as EH

from src import fun_bbi as bbi
from src.params import IDX, eeg_bands, eeg_channels, com_vars, \
    max_m_se, tau_bbi, radii

# define paths & load data ----------
PATH_DATA = './data/npy/'
PATH_OUT = './data/'
PATH_FIGS = './output/dat/'

# load data... 
data = bbi.import_npz(PATH_DATA)
COM, EEG = data['COM_128'], data['EEG_128']
EEG_AGG = bbi.agg_bands(EEG, eeg_bands, eeg_channels)



# Cross SampEn ------------------------------------------------
# calculate sample entropy between channel-wise EEG and COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
nx = (len(EEG.keys())*len(eeg_channels)*3*len(com_vars)*len(eeg_bands))
cols = ['id', 'band', 'channel_num', 'channel_label','com_var', 'radius', 
        'm0', 'm1', 'm2', 'm3', 'm4', 'm5']
xse_opt = np.zeros((nx, len(cols)))
xse_opt = pd.DataFrame(xse_opt, columns=cols)

row = 0
for xid in IDX:
    for lobe in eeg_channels:
        for channel in eeg_channels[lobe]:
            for band in eeg_bands:
                for xvar in com_vars:
                    print(xid, lobe, '\t', band, channel, '\t', xvar)
                    
                    # import COM and EEG data
                    tcom, com, teeg, eeg = bbi.align_data(COM, EEG, xid, band, lobe, channel, xvar)

                    # normalize
                    com = bbi.normalize(com, scale_max=False)
                    eeg = bbi.normalize(eeg, scale_max=False)
                    # combine com and eeg data into array for xssampen 
                    ts = np.stack([np.array(eeg), np.array(com)])

                    for r in radii:
                        se = EH.XSampEn(ts, m=max_m_se, r=r, tau=tau_bbi)[0]

                        # index data
                        indemo = [xid, band, channel, eeg_channels[lobe][channel], xvar, r]
                        xse_opt.loc[row, cols[:6]] = indemo
                        xse_opt.loc[row, cols[6:]] = se

                        row += 1

xse_opt
# xse_opt.to_excel(PATH_OUT + 'nld/05_bbi__xse.xlsx', index=False)


# calculate sample entropy between aggregated EEG and COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
nx = (len(IDX.keys())*len(eeg_channels)*len(com_vars)*len(eeg_bands))
cols = ['id', 'lobe', 'band', 'com_var', 'radius',
        'm0', 'm1', 'm2', 'm3', 'm4', 'm5']
xse_opt2 = np.zeros((nx, len(cols)))
xse_opt2 = pd.DataFrame(xse_opt2, columns=cols)

row = 0
for xid in IDX:
    for lobe in eeg_channels:
        for band in eeg_bands:
            for xvar in com_vars:
                print(xid, lobe, '\t', band, '\t', xvar)
                channel = ''
                # import COM and EEG data
                tcom, com, teeg, eeg = bbi.align_data(COM, EEG_AGG, xid, band, lobe, channel, xvar)

                # normalize
                com = bbi.normalize(com, scale_max=False)
                eeg = bbi.normalize(eeg, scale_max=False)
                # combine com and eeg data into array for xssampen 
                ts = np.stack([np.array(eeg), np.array(com)])

                for r in radii:
                    se = EH.XSampEn(ts, m=max_m_se, r=r, tau=tau_bbi)[0]

                    # index data
                    indemo = [xid, lobe, band, xvar, r]
                    xse_opt2.loc[row, cols[:5]] = indemo
                    xse_opt2.loc[row, cols[5:]] = se

                    row += 1

xse_opt2
# xse_opt2.to_excel(PATH_OUT + 'nld/05_bbi__xse_agg.xlsx', index=False)


# end
