# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script performed recurrence quantificaiton analysis
#     on the univarate time series... 
#   - comparisons of alternative parameters is not performed here
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import numpy as np
import pandas as pd 

from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Classic
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius

from src import fun_bbi as bbi

from src.params import IDX, eeg_bands, eeg_channels, com_vars, rqa_vars, \
    m_eeg, m_com, tau_eeg, tau_com, r_rqa, theiler


# define paths
PATH_DATA = './data/npy/'
PATH_OUT = './data/'

# load data...
data = bbi.import_npz(PATH_DATA)
COM, EEG = data['COM_128'], data['EEG_128']
EEG_AGG = bbi.agg_bands(EEG, eeg_bands, eeg_channels) # agg data 



# RQA -----------------------------------------------------
# calculate recurrence for aggregated EEG bands
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
nx = (len(EEG.keys())*len(eeg_channels)*len(eeg_bands))

rqa_eeg = np.zeros((nx, 3+len(rqa_vars)))
rqa_eeg = pd.DataFrame(rqa_eeg, columns=('id', 'lobe', 'band') + rqa_vars)

row = 0
for xid in EEG:
    for lobe in eeg_channels:
        for ii, band in enumerate(eeg_bands):
            print(xid, lobe, '\t', band)
            channel, xvar = '', ''

            x = EEG_AGG[xid][lobe]['time']
            eeg = bbi.normalize(EEG_AGG[xid][lobe][band])

            # rqa
            radius = r_rqa*eeg.std()
            ts = TimeSeries(eeg, embedding_dimension=m_eeg, time_delay=tau_eeg)
            settings = Settings(ts,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(radius),
                                similarity_measure=EuclideanMetric,
                                theiler_corrector=theiler)
            # recurrence quantification 
            rqa_comp = RQAComputation.create(settings, verbose=False)
            rqa = rqa_comp.run()
            rqa.min_diagonal_line_length = 3
            rqa.min_vertical_line_length = 3
            rqa.min_white_vertical_line_length = 3

            # pull results 
            det = rqa.determinism
            rec = rqa.recurrence_rate
            lam = rqa.laminarity
            dline = rqa.longest_diagonal_line
            vline = rqa.longest_vertical_line
            dent = np.nan
            vent = np.nan

            out = [xid, lobe, band, det, rec, lam, dline, vline, dent, vent]
            rqa_eeg.iloc[row, :] = out

            row +=1

rqa_eeg
# rqa_eeg.to_excel(PATH_OUT + 'nld/04_eeg__rqa.xlsx', index=False)


# COM ------------------------------------------------------
# initiate dataframe for analysis of COM data 
nx = len(IDX)*len(com_vars)
rqa_com = np.zeros((nx, 2+len(rqa_vars)))
rqa_com = pd.DataFrame(rqa_com, columns=('id', 'com_var') + rqa_vars)

row = 0
for ii, xid in enumerate(IDX):
    for k, xvar in enumerate(com_vars.keys()):
        print(xid, xvar)
        
        x = COM[xid]['time']
        com = bbi.normalize(COM[xid][xvar])

        # rqa
        radius = 2*r_rqa*com.std()  # adjusted r for COM data
        ts = TimeSeries(com, embedding_dimension=m_com, time_delay=tau_com)
        settings = Settings(ts,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(radius),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=theiler)
        # recurrence quantification 
        rqa_comp = RQAComputation.create(settings, verbose=False)
        rqa = rqa_comp.run()
        rqa.min_diagonal_line_length = 3
        rqa.min_vertical_line_length = 3
        rqa.min_white_vertical_line_length = 3

        # pull results 
        det = rqa.determinism
        rec = rqa.recurrence_rate
        lam = rqa.laminarity
        dline = rqa.longest_diagonal_line
        vline = rqa.longest_vertical_line
        dent = np.nan
        vent = np.nan

        out = [xid, xvar, det, rec, lam, dline, vline, dent, vent]
        rqa_com.iloc[row, :] = out

        row +=1

rqa_com
# rqa_com.to_excel(PATH_OUT + 'nld/04_com__rqa.xlsx', index=False)

# end
