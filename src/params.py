# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this file contains a variety of parameters needed 
#     throughout the processing of the EEG and COM data.
#
# ---------------------------------------------------------

import numpy as np


# smapling frequency of data
# - must match EEG and COM data
fs = 128

# specifc subjects for analysis
# - used in manuscripts plots and some of the 
#   analyses to shorten runtime
IDX = {'H05': 'S1', 'H07': 'S2', 'C01': 'S3'}

# EEG bands
eeg_bands = ('SCO', 'alpha', 'beta', 'gamma')

# total EEG channels 
N_CHANNELS = 32 

# EEG channels (by hemisphere) used in analysis
eeg_channels = {'left': {8: 'C3', 
                         6: 'FC5', 
                         11: 'CP5'},
                'right': {25: 'C4', 
                          28: 'FC6', 
                          22: 'CP6'}}

# center of mass (COM)
com_vars = {'ap__m_s2': 'AP',
            'ml__m_s2': 'ML', 
            'vt__m_s2': 'VT', 
            'res__m_s2': 'RES'}


# max lags for acf and ami -------------------
mlag_acf = int(fs*10)
mlag_ami = int(fs/2)

# time delays for analyses --------------------
tau_eeg = 16    # 
tau_com = 16    # 
tau_bbi = 16    # 

# max embedding dimension ---------------------
max_m_fnn = 10      # for fnn
max_m_se = 5        # for smaple entropy
m_eeg = 3           #
m_com = 3           #
m_bbi = 3           # 
r_se = 0.2          # radius for sample entropy 

# radii for sampen ----------------------------
radii = np.arange(0.10, 0.4, 0.05)

# embedding labels 
se_labels = {'m1': 'm=1', 'm2': 'm=2', 'm3': 'm=3', 
             'm4': 'm=4', 'm5': 'm=5'}

# RQA and cRQA parameters
rqa_vars = ('det', 'rec', 'lam', 'long_dline', 
            'long_vline', 'entropy_dlines', 'entropy_vlines')
r_rqa = 0.23
theiler = 1

# time-range ---------------------------------
# - used in state space reconstruction and
#   the calculation of false nearest neighbors 
#   to shorten runtime (validated)
trange = (45, 75)


# end 
