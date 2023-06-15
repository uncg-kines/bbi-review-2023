# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script calcualtes cross-RQA for COM and EEG data
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import EntropyHub as EH

from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Cross
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius

from src import fun_bbi as bbi
from src.params import eeg_bands, eeg_channels, com_vars, rqa_vars, \
    m_bbi, tau_bbi, r_rqa, theiler


# define paths & load data ----------
PATH_DATA = './data/npy/'
PATH_OUT = './data/'
PATH_FIGS = './output/dat/'

# load data... 
data = bbi.import_npz(PATH_DATA)
COM, EEG = data['COM_128'], data['EEG_128']
EEG_AGG = bbi.agg_bands(EEG, eeg_bands, eeg_channels)



# CRQA -----------------------------------------------------
# calculate cross-recurrence between channel-wise EEG and COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
nx = (len(EEG)*len(eeg_channels)*3*len(com_vars)*len(eeg_bands))
rqa_vars = ('det', 'rec', 'lam', 'long_dline', 
            'long_vline', 'entropy_dlines', 'entropy_vlines')

rqa_out = np.zeros((nx, 4+len(rqa_vars)))
rqa_out = pd.DataFrame(rqa_out, columns=('id', 'band', 'channel', 'com_var') + rqa_vars)

row = 0
for xid in EEG:
    for lobe in eeg_channels:
        fig, ax = plt.subplots(len(eeg_bands), 2, figsize=(5, 7), sharey=True)
        plt.rcParams.update({'font.size': 8})

        for ii, band in enumerate(eeg_bands):
            for j, xvar in enumerate(com_vars):
                for channel in eeg_channels[lobe]:
                    print(xid, lobe, '\t', band, channel, '\t', xvar)

                    # align COM and EEG data 
                    tcom, com, teeg, eeg = bbi.align_data(
                        COM, EEG, xid, band, lobe, channel, xvar)
                    
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
                    # recurrence quantification 
                    crqa_comp = RQAComputation.create(settings, verbose=False)
                    crqa = crqa_comp.run()
                    crqa.min_diagonal_line_length = 3
                    crqa.min_vertical_line_length = 3
                    crqa.min_white_vertical_line_length = 3

                    # pull results 
                    det = crqa.determinism
                    rec = crqa.recurrence_rate
                    lam = crqa.laminarity
                    dline = crqa.longest_diagonal_line
                    vline = crqa.longest_vertical_line
                    dent = np.nan
                    vent = np.nan

                    out = [xid, band, int(channel), xvar, det, rec, lam, dline, vline, dent, vent]
                    rqa_out.iloc[row, :] = out

                    row +=1

rqa_out
# rqa_out.to_excel(PATH_OUT + 'nld/05_bbi__crqa.xlsx', index=False)


# calculate cross-recurrence between aggregated EEG and COM data 
#   - EEG channels (Left: C3, FC5, CP5; Right: C4, FC6, CP6)
#   - COM data [AP, ML, VT, RES(AP, ML, VT)]
nx = (len(EEG.keys())*len(eeg_channels)*len(com_vars)*len(eeg_bands))
rqa_vars = ('det', 'rec', 'lam', 'long_dline', 
            'long_vline', 'entropy_dlines', 'entropy_vlines')

rqa_out2 = np.zeros((nx, 4+len(rqa_vars)))
rqa_out2 = pd.DataFrame(rqa_out2, columns=('id', 'lobe', 'band', 'com_var') + rqa_vars)

row = 0
for xid in EEG:
    for lobe in eeg_channels:
        for ii, band in enumerate(eeg_bands):
            for j, xvar in enumerate(com_vars):
                print(xid, lobe, '\t', band, '\t', xvar)
                channel = ''

                # align COM and EEG data 
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
                # recurrence quantification 
                crqa_comp = RQAComputation.create(settings, verbose=False)
                crqa = crqa_comp.run()
                crqa.min_diagonal_line_length = 3
                crqa.min_vertical_line_length = 3
                crqa.min_white_vertical_line_length = 3

                # pull results 
                det = crqa.determinism
                rec = crqa.recurrence_rate
                lam = crqa.laminarity
                dline = crqa.longest_diagonal_line
                vline = crqa.longest_vertical_line
                dent = np.nan
                vent = np.nan

                out = [xid, lobe, band, xvar, det, rec, lam, dline, vline, dent, vent]
                rqa_out2.iloc[row, :] = out

                row +=1

rqa_out2
# rqa_out2.to_excel(PATH_OUT + 'nld/05_bbi__crqa_agg.xlsx', index=False)

# end
