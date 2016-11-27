import pandas as pd
import os
from Functions import feature_prune

print "\n"
datafile = raw_input("Pandas dataframe to open: ")
dataframe = pd.read_hdf(os.path.join('Dataframes', "%s.h5" % datafile))

print "\n"
print "Dataframe description:"
with open(os.path.join('Data_desc', '%s.txt' % datafile)) as f:
    print f.read()
print "\n"

# Depending on the RF type, removes one of the target observations.
prune_style = raw_input("Classisfication or Regression (c/r): ")

# filename
filename = raw_input("Pruned file name: ")

# General features which give the positions of critical circuits and ecg probes.
general_features = ['Crit Position 0', 'Crit Position 1', 'Probe Position', 'Distance 0', 'Distance 1',
                    'Unit Vector X 0', 'Unit Vector X 1', 'Unit Vector Y 0',
                    'Unit Vector Y 1', 'Theta 0', 'Theta 1', 'Target 0', 'Target 1',
                    'Nearest Crit Position']

feature_prune(dataframe, general_features)

# Prunes away the Largest FT Mag/Freq as they have little impact on feature importance.
feature_prune(dataframe, ['Largest FT Mag %s' % x for x in range(1, 10)])
feature_prune(dataframe, ['Largest FT Freq %s' % x for x in range(1, 10)])


if prune_style == 'c':
    feature_prune(dataframe, ['True Distance'])

if prune_style == 'r':
    feature_prune(dataframe, ['True Target'])

dataframe.to_hdf(os.path.join('Dataframes', "%s.h5" % filename), 'w')
