# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this file import the EEG and COM data and stores it as an *.npz 
#     file for use in subsequent scripts 
#
# ---------------------------------------------------------
# ---------------------------------------------------------

import os
import numpy as np

from src import fun_bbi as bbi


# import orig data -------------------------
PATH_LIST = ('./data/orig/eeg/', './data/orig/com/')
PATH_OUT = './data/npy/'

FILE_LIST = []
for ii, path in enumerate(PATH_LIST):
    FILE_LIST.append(os.listdir(path))

# processed EEG data and raw COM data
EEG_128 = bbi.import_eeg(PATH_LIST[0], FILE_LIST[0])
COM_128 = bbi.import_com(PATH_LIST[1], FILE_LIST[1])

# export data ------------------------
data = {'EEG_128': EEG_128,
        'COM_128': COM_128}

# save data 
for dkey in data.keys():
    if not os.path.exists(PATH_OUT + dkey):
        os.mkdir(PATH_OUT + dkey)
    for pkey in data[dkey].keys():
        np.savez(PATH_OUT + dkey + '/' + pkey, **data[dkey][pkey])


# end 
