import h5py as h5
import numpy as np

filename = raw_input("h5 filename: ")
testfile = h5.File(filename, 'r')
def get_group(number):
    group = testfile.get('Index: ' + str(number))
    return group
def get_cp(number):
    return np.array(get_group(number)['Crit Position'])
def get_probes(number):
    return np.array(get_group(number)['Probe Positions'])
def get_ecg(number):
    return np.array(get_group(number)['ECG'])
