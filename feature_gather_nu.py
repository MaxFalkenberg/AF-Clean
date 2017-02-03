import pandas as pd
import os
from Functions import feature_extract_nu
import numpy as np
import h5py
from numpy.fft import rfft
from Functions import print_progress
# SingleSource_ECGdata_Itt1000_P60.h5

filename = raw_input("Training data to convert into pandas dataframe: ")

trainfile = h5py.File('%s.h5' % filename, 'r')
group = trainfile.get('Index: 0')
cp_number = len(np.atleast_1d(np.array(group['Crit Position'])))
feature_grid = None  # Numpy Zero array
index_num = None  # Integer
probe_num = trainfile['Index: 0']['Probe Positions'].shape[0]

largest_ft_freq_columns = ['FT Freq %s' % x for x in range(0, 4)]
largest_ft_mag_columns = ['FT Mag %s' % x for x in range(0, 4)]
largest_ft_rel1_mag_columns = ['FT Rel 1 Mag %s' % x for x in range(0, 4)]
largest_ft_rel2_mag_columns = ['FT Rel 2 Mag %s' % x for x in range(0, 4)]
largest_ft_rel3_mag_columns = ['FT Rel 3 Mag %s' % x for x in range(0, 4)]
crit_pos = ['Crit Position %s' % x for x in range(cp_number)]
dist = ['Distance %s' % x for x in range(cp_number)]
vecx = ['Vector X %s' % x for x in range(cp_number)]
vecy = ['Vector Y %s' % x for x in range(cp_number)]
uvecx = ['Unit Vector X %s' % x for x in range(cp_number)]
uvecy = ['Unit Vector Y %s' % x for x in range(cp_number)]
theta = ['Theta %s' % x for x in range(cp_number)]
targ = ['Target %s' % x for x in range(cp_number)]
#'Entropy','Hurst Exponent','Correlation Dimension','Detrended Fluctuation Analysis',

columns = ['Covariance','Mean','Skewness','Kurtosis','Max Value', 'Min Value', 'Minmax Diff', 'Max Arg', 'Min Arg','Minmax Half','Arg Half','Half Ratio','STD Full','STD Premax','STD Minmax','STD Postmin', 'Positive Sample Intensity','Negative Sample Intensity','Grad Minmax Mean', 'Sample Intensity', 'Sample Length', 'Grad Max','Grad Min',
           'Grad Diff', 'Grad Argmax', 'Grad Argmin', 'Grad Argdiff', 'First Station'] \
          + largest_ft_freq_columns + largest_ft_mag_columns + largest_ft_rel1_mag_columns + largest_ft_rel2_mag_columns + largest_ft_rel3_mag_columns +\
           crit_pos + ['Probe Position'] + dist + vecx + vecy +uvecx + uvecy + theta + targ + ['Nearest Crit Position', 'Nu']

memory_condition = False

for i in range(len(trainfile.keys())):
    if len(trainfile['Index: %s' % i]) < 3:
        print("---Warning---")
        print("Index Issue: %s" % i)
        print("Number of Full Iterations: %s" % (i-1))
        index_num = i-1
        feature_grid = np.zeros((probe_num * (i-1), len(columns)))
        memory_condition = True

if memory_condition is False:
    print("Number of Full Iterations: %s" % len(trainfile.keys()))
    index_num = len(trainfile.keys())
    feature_grid = np.zeros((probe_num * index_num, len(columns)))

print feature_grid.shape

count = 0
i = 0


for index in range(index_num):
    group = trainfile.get('Index: %s' % index)
    cp = np.array(group['Crit Position'])
    probes = np.array(group['Probe Positions']).astype(int)
    nu = np.array(group['Nu'])
    ecg_vals = np.array(group['ECG'])
    ecg_0 = ecg_vals[0]
    fft_0 = rfft(ecg_0)
    i += 1
    print_progress(i, index_num, prefix='Progress:', suffix='Complete', bar_length=50)
    for number in range(probe_num):
        feature_grid[count] = feature_extract_nu(number, ecg_vals=ecg_vals, cp=cp, probes=probes, nu = nu)
        count += 1

df_index = range(index_num * probe_num)
df = pd.DataFrame(data=feature_grid, index=df_index, columns=columns)
df.to_hdf("%s_df.h5" % filename, 'w')
