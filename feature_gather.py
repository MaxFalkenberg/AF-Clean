import pandas as pd
import time
from Functions import feature_extract
import numpy as np
import h5py
from numpy.fft import rfft
# from numpy.fft import irfft
# SingleSource_ECGdata_Itt1000_P60.h5

filename = raw_input("Training data to convert into pandas dataframe: ")
trainfile = h5py.File('%s.h5' % filename, 'r')
feature_grid = np.zeros((400000, 43))

largest_ft_freq_columns = ['Largest FT Freq %s' % x for x in range(1, 11)]
largest_ft_mag_columns = ['Largest FT Mag %s' % x for x in range(1, 11)]
largest_ft_rel_mag_columns = ['Largest FT Rel Mag %s' % x for x in range(1, 11)]

columns = ['Max Value', 'Min Value', 'Minmax Diff', 'Sample Intensity', 'Sample Length', 'Grad Max', 'Grad Min',
           'Grad Diff', 'Grad Argmax', 'Grad Argmin', 'Grad Argdiff'] + largest_ft_freq_columns + \
          largest_ft_mag_columns + largest_ft_rel_mag_columns + ['Largest Sum', 'Crit Position',
                                                                 'Probe Position', 'Distance', 'Target']

count = 0

start_time1 = time.time()
for index_num, index in enumerate(trainfile.iterkeys()):
    group = trainfile.get('%s' % index)
    cp = np.array(group['Crit Position'])
    probes = np.array(group['Probe Positions']).astype(int)
    ecg_vals = np.array(group['ECG'])
    ecg_0 = ecg_vals[0]
    fft_0 = rfft(ecg_0)
    for number in range(400):
        feature_grid[count] = feature_extract(number, ecg_vals=ecg_vals, cp=cp, probes=probes)
        count += 1
print("--- Simulation: %s seconds ---" % (time.time() - start_time1))

df_index = range(400000)
df = pd.DataFrame(data=feature_grid, index=df_index, columns=columns)
df.to_hdf("%s_df.h5" % filename, 'w')
