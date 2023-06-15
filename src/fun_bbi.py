# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------
# NOTES: 
#   - this script contains a varity of functions that are
#     called in the brain-body analyses... 
#   - the included functions are specific to importing, exporting,
#     processing, etc.
#   - all functions associated with nonlinear dynamics are provided 
#     in '~/src/nonlinear_dynamics.py'
#
# ---------------------------------------------------------
# ---------------------------------------------------------


print(''' 
      \n\n
      # -------------------------------------------------
      author: nate berry (nberry11 at gmail dot com)
      citation: 
        <tbd>
      # -------------------------------------------------
      this script contains a variety of functions used in
      the processing of data presented in the above citation... 
      
      we have worked diligently to make sure that there are
      no errors within this code and that everything is 
      reproducible, without error. if you come across an error, 
      please let us know. 

      # -------------------------------------------------
      \n\n 
      ''')


import os
import numpy as np
import pandas as pd 
import scipy.io as sio

from scipy.signal import butter, filtfilt


def align_data(dCOM, dEEG, xid, band, lobe, channel, xvar, tstart=10):
    """ align COM and EEG data"""
    # select data 
    tcom = dCOM[xid]['time']
    com = dCOM[xid][xvar]

    if len(dEEG[xid]) != 2:
        teeg = dEEG[xid]['time'].reshape([1, len(dEEG[xid]['time'])])[0]
        eeg = dEEG[xid][band][:, channel-1]
    else: 
        teeg = dEEG[xid][lobe]['time']
        eeg = dEEG[xid][lobe][band]

    # determine baseline minimum (should be from eeg)
    if min(tcom)!=min(teeg):
        raise Exception('times do not match...something is off')

    # flag corresponding time series
    tmax = min(max(tcom), max(teeg))
    f1 = (tcom > tstart) & (tcom < tmax)
    f2 = (teeg > tstart) & (teeg < tmax)

    # subset time and time series 
    tcom, com = tcom[f1], com[f1]
    teeg, eeg, = teeg[f2], eeg[f2]

    return tcom, com, teeg, eeg


def agg_bands(eeg_data, eeg_bands, eeg_channels):
    aggdat = {}
    for xid in eeg_data:
        ldat = {}
        for lobe in eeg_channels:
            bdat = {}
            for j, band in enumerate(eeg_bands):
                # adjust for zero indexing
                channels = np.array(list(eeg_channels[lobe]))-1
                bdat[band] = eeg_data[xid][band][:, channels].mean(axis=1)

            bdat['time'] = eeg_data[xid]['time'].reshape(-1)
            ldat[lobe] = bdat
        aggdat[xid] = ldat

    return aggdat


def downsample(data, fs=128, fsn=32):
    w = fs / fsn

    if w.is_integer():
        x, y = data[0], data[1]
        x, y = x[::int(w)], y[::int(w)]
    else: 
        print('fs/fsn must be an integer')

    return x, y


def export_npz(data, path_out):
    """export data to npz file for use later"""
    for dkey in data:
        if not os.path.exists(path_out + dkey):
            os.mkdir(path_out + dkey)
        for pkey in data[dkey]:
            np.savez(path_out + dkey + '/' + pkey, **data[dkey][pkey])


def import_com(path, file_list, verbose=True):
    """import center of mass data"""
    vdict = {'Time_s': 'time',
             'CoM_LinAcc_AP_m_s2': 'ap__m_s2',
             'CoM_LinAcc_ML_m_s2': 'ml__m_s2',
             'CoM_LinAcc_VT_m_s2': 'vt__m_s2',
             'CoM_LinAcc_Res_m_s2': 'res__m_s2',
             'CoM_AngVel_AP_deg_s': 'ap__deg_s',
             'CoM_AngVel_ML_deg_s': 'ml__deg_s',
             'CoM_AngVel_VT_deg_s': 'vt__deg_s'}

    data_com = {}
    for file in file_list:
        if file[0] == '.':
            continue

        if verbose:
            print(file)

        xid = file.split('_')[0]
        d = pd.read_excel(path + file)

        # subtract out gravity from vertical 
        gravity = 9.81
        d.CoM_LinAcc_VT_m_s2 = d.CoM_LinAcc_VT_m_s2 - gravity

        # calculate resultant (accel)
        d['CoM_LinAcc_Res_m_s2'] = np.sqrt(
            d.CoM_LinAcc_AP_m_s2**2 
            + d.CoM_LinAcc_ML_m_s2**2
            + d.CoM_LinAcc_VT_m_s2**2)

        for j in range(1, d.shape[1]):
            d.iloc[:, j] = lowpass_butter(d.iloc[:, j], fc=8, fs=128, order=4)

        data_id = {}
        for var in vdict:
            data_id[vdict[var]] = d[var].to_numpy()

        data_com[xid] = data_id

    return data_com


def import_eeg(path, file_list, subtype='orig', verbose=True):
    """import EEG data. converts from .mat file to dictionary"""
    data_eeg = {}
    for file in file_list:
        if file[0] == '.':
            continue

        if verbose:
            print(file)

        xid = file.split('_')[0]
        matobj = sio.loadmat(path + file)['EEGts']
        bands = matobj[0, 0][subtype].dtype.names

        data_id = {}
        for band in bands:
            dband = matobj[0, 0][subtype][0, 0][band]
            if band=='time':
                dband = dband.T[0]
            else:
                dband = dband.T

            data_id[band] = dband

        data_eeg[xid] = data_id

    return data_eeg


def import_npz(path, data_names=('COM_128', 'EEG_128'), verbose=True):
    """import .npz files"""
    list_folders = [f for f in os.listdir(path) if not f.startswith('.')]

    data = {}
    for folder in list_folders:
        list_files = os.listdir(path + folder)
        data_id = {}
        for file in list_files:
            if '.DS' in file:
                continue

            xid = file.split('.')[0]
            if verbose: print(folder, '\t', xid)

            data_id[xid] = dict(np.load(path + folder + '/' + file, 
                                        allow_pickle=True))
        data[folder] = data_id
    
    out = {}
    for dname in data_names:
        out[dname] = data[dname]

    return out


def lowpass_butter(xn, fc, fs, order):
    nc = fc / (0.5 * fs) # normal cutoff

    b, a = butter(order, nc, btype='low', analog=False)
    wn = filtfilt(b, a, xn)

    return wn


def normalize(ts, yambda=1, scale_max=True):
    ts = np.array(ts)

    if scale_max:
        ts = (ts-ts.mean()) / (ts.std())
        ts = ts/np.abs(max(ts))
    else: 
        ts = (ts-ts.mean()) / (ts.std() * yambda)

    return ts


