%% 
% This script reads in any 'cleaned' EEG data 
% that are stored as .mat files and EEGlab 'struct' format
% in $SOURCEDATAPATH.This is a conversion fro 
%
% ver 0.1 28APR23 dcm
%%
% Set pathnames and generate list of all files 
PROJPATH = '~/bbi-review-2023';
DERIVATIVESPATH = sprintf('%s/data/orig/eeg/',PROJPATH) ;
SOURCEDATAPATH = sprintf('%s/path/to/clean/eeg/data/',PROJPATH) ; ;
FTPATH = '/path/to/fieldtrip'
addpath(genpath(SYNKPROJPATH))
addpath(FTPATH)
ft_defaults
cd(SOURCEDATAPATH)
list = dir("*.mat");
fnames = {list.name};

% Modify band names/limits (Hz)
bands = {'SCO','alpha','beta','gamma'};
lo_freq = [1 8 14 30];
hi_freq = [7.5 13.5 29.5 50];

for j = 1:length(fnames)
    cleandat = fnames{j};
    load(sprintf('%s',cleandat))
    % if you want to name the output with a subject id
    % then modify these lines to extract/name that variable
    % with each loop
    fnameparts = split(cleandat,'_');
    sub = convertCharsToStrings(fnameparts{1});
    %replace cleanEEG with the name of your cleaned EEG struct
    data_eeg = eeglab2fieldtrip(cleanEEG,'raw')
    data_eeg.dimord = 'chan_time';
    %resample if necessary
    cfg = [];
    cfg.resamplefs = 256;
    cfg.detrend = 'no';
    data_eeg_resamp = ft_resampledata(cfg, data_eeg)
    %read in as continuous data without furter preprocessing
    cfg = [];
    cfg.continuous = 'yes';
    data_eeg_preproc = ft_preprocessing(cfg,data_eeg_resamp);
    %this will determine resolution of new timefrequency data
    lgth = data_eeg_preproc.sampleinfo(2);
    begsample = [1:2:lgth-512]';
    endsample = [513:2:lgth]';
    cfg = [];
    cfg.trl = [begsample,endsample,[0:2:lgth-513]'];
    data_eeg_preproc_seg = ft_redefinetrial(cfg, data_eeg_preproc);
    %use multitapers to perform the transform 
    cfg           = [];
    cfg.method    = 'mtmfft';
    cfg.taper     = 'dpss';
    cfg.output    = 'pow';
    cfg.pad       = 'nextpow2';
    cfg.foilim    = [1 50];
    cfg.keeptrials= 'yes';
    cfg.tapsmofrq= 3;
    data_eeg_freqproc = ft_freqanalysis(cfg, data_eeg_preproc_seg);
    %manually recreate timestamps
    time = [];
    for j = 1:size(data_eeg_preproc_seg.time,2)
        timestamps = cell2mat(data_eeg_preproc_seg.time(j));
        time(j) = timestamps(1);
    end
    %convert to chan x freq x time 
    data_eeg_freqproc_cont           = data_eeg_freqproc;
    data_eeg_freqproc_cont.powspctrm = permute(data_eeg_freqproc.powspctrm, [2, 3, 1]);
    data_eeg_freqproc_cont.dimord    = 'chan_freq_time'; 
    data_eeg_freqproc_cont.time      = time;           
    %for each band, identify the resulting frequencies that are nearest
    %the hi/lo limits (Hz) and average signals between those columns
    for y = 1:length(bands)
        fqlo = lo_freq(y);
        fqhi = hi_freq(y);
        [~,lo_idx]=min(abs(data_eeg_freqproc_cont.freq-fqlo));
        [~,hi_idx]=min(abs(data_eeg_freqproc_cont.freq-fqhi));
        EEGts.orig.(bands{y}) = squeeze(mean(data_eeg_freqproc_cont.powspctrm(:,[lo_idx:hi_idx],:),2));
    end
    %Copy time stamps over
    EEGts.orig.time = data_eeg_freqproc_cont.time';
    save(sprintf('%s/%s_EEGts.mat',DERIVATIVESPATH,sub),"EEGts")
end
