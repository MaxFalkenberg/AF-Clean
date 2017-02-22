import pandas as pd
import os
from Functions import feature_extract_keras
import numpy as np
import h5py
from numpy.fft import rfft
from Functions import print_progress
# SingleSource_ECGdata_Itt1000_P60.h5

filename = raw_input("Training data to convert into pandas dataframe: ")

trainfile = h5py.File('%s.h5' % filename, 'r')
group = trainfile.get('Index: 0')
ecg_vals = np.array(group['ECG'])
cp = np.array(group['Crit Position'])
probes = np.array(group['Probe Positions']).astype(int)
nu = np.array(group['Nu'])
cp_number = len(np.atleast_1d(np.array(group['Crit Position'])))
feature_grid = None  # Numpy Zero array
probe_num = trainfile['Index: 0']['Probe Positions'].shape[0]


index_num = len(trainfile.keys())
count = 0
i = 0

sample_len = len(feature_extract_keras(0, ecg_vals=ecg_vals, cp=cp, probes=probes, nu = nu))
columns = ['SampleNumber %s' % x for x in range(probe_num * index_num)]
time = ['Time %s' % x for x in range(sample_len)]
row_index = ['Nu', 'X Vector', 'Y Vector'] + time
# grid = np.zeros((sample_len,len(columns)))
# print np.shape(grid), len(columns)
lengths = np.zeros(probe_num * index_num)


for index in range(index_num):
    group = trainfile.get('Index: %s' % index)
    cp = np.array(group['Crit Position'])
    probes = np.array(group['Probe Positions']).astype(int)
    nu = np.array(group['Nu'])
    # pn = np.array(group['Probe Number'])
    ecg_vals = np.array(group['ECG'])
    ecg_0 = ecg_vals[0]
    fft_0 = rfft(ecg_0)
    i += 1
    print_progress(i, index_num, prefix='Progress:', suffix='Complete', bar_length=50)
    for number in range(probe_num):
        lengths[count] = len(feature_extract_keras(number, ecg_vals=ecg_vals, cp=cp, probes=probes, nu = nu))
        count += 1

count = 0
i = 0
min_length = int(np.max(lengths))

sample_len = len(feature_extract_keras(0, ecg_vals=ecg_vals, cp=cp, probes=probes, nu = nu))
columns = ['SampleNumber %s' % x for x in range(probe_num * index_num)]
time = ['Time %s' % x for x in range(sample_len - 3)]
row_index = ['Nu', 'X Vector', 'Y Vector'] + time
grid = np.zeros((int(np.max(lengths)),len(columns)))
print np.shape(grid), len(columns)

for index in range(index_num):
    group = trainfile.get('Index: %s' % index)
    cp = np.array(group['Crit Position'])
    probes = np.array(group['Probe Positions']).astype(int)
    nu = np.array(group['Nu'])
    # pn = np.array(group['Probe Number'])
    ecg_vals = np.array(group['ECG'])
    ecg_0 = ecg_vals[0]
    fft_0 = rfft(ecg_0)
    i += 1
    print_progress(i, index_num, prefix='Progress:', suffix='Complete', bar_length=50)
    for number in range(probe_num):
        grid[:,count] = feature_extract_keras(number, ecg_vals=ecg_vals, cp=cp, probes=probes, nu = nu, min = min_length)
        count += 1

df_index = range(index_num * probe_num)
df = pd.DataFrame(data=grid, index=row_index, columns=columns)
df.to_hdf("%s_keras.h5" % filename, 'w')
