"""
Temporary file as couldn't change pandas column names
Distance 0 --> Distance in this file.
"""

import pandas as pd

print "\n"
datafile = raw_input("Pandas dataframe to open: ")
dataframe = pd.read_hdf("%s.h5" % datafile)

print "\n"
print "Dataframe description:"
with open('%s.txt' % datafile) as f:
    print f.read()
print "\n"

# Depending on the RF type, removes one of the target observations.
observation_style = raw_input("Classisfication/Regression (c/r): ")
prune = raw_input("Prune features? (y/n): ")

# filename
filename = raw_input("Saved file name: ")

# General features which give the positions of critical circuits and ecg probes.
probe_features = ['Crit Position', 'Crit Position 0', 'Crit Position 1', 'Probe Position', 'Distance 1',
                  'Unit Vector X', 'Unit Vector X 0', 'Unit Vector X 1', 'Unit Vector Y', 'Unit Vector Y 0',
                  'Unit Vector Y 1', 'Theta', 'Theta 0', 'Theta 1', 'Target 1', 'Nearest Crit Position']

# Deletes features from the dataframe that are in probe_features
all_features = list(dataframe.columns)
for feature in probe_features:
    if feature in all_features:
        del dataframe['%s' % feature]

# # Prunes away the Largest FT Mag/Freq as they have little impact on feature importance.
# if prune == 'y':
#     feature_prune(dataframe, ['Largest FT Mag %s' % x for x in range(1, 10)])
#     feature_prune(dataframe, ['Largest FT Freq %s' % x for x in range(1, 10)])

if observation_style == 'c':
    del dataframe['Distance 0']

if observation_style == 'r':
    del dataframe['Target 0']

dataframe.to_hdf("%s.h5" % filename, 'w')
