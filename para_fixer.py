"""
File should convert _para.h5 files into the right format for use in open.

SAVE THE PARA FILES BEFORE YOU TRY THIS AS IT WILL OVERWRITE THEM.
"""

import h5py
print '\n'
check = 0
file_convert = raw_input("Enter para file that needs converting (e.g delta_analysis_m1) : ")
while check == 0:
    print "-------------------------------------------"
    print "File being converted: %s" % file_convert
    yes_no = raw_input("Is this the correct file? [y/n]: ")
    print "-------------------------------------------"
    if yes_no == 'y':
        check = 1
        print '\n'
    else:
        print '\n'
        file_convert = raw_input("Enter _para.h5 file that needs converting: ")

with h5py.File('%s.h5' % file_convert, 'a') as hf:
    print('List of arrays in this file: \n', hf.keys())
    groups = hf.get('parameters')
    groups['Delta'] = groups['delta']
    del groups['delta']
    groups['Nu'] = groups['nu']
    del groups['nu']
    groups['Pulse Rate'] = groups['pulse_rate']
    del groups['pulse_rate']
    groups['Refractory Period'] = groups['rp']
    del groups['rp']
    groups['Epsilon'] = groups['epsilon']
    del groups['epsilon']
    groups['Simulation Size'] = 100000  # Assuming that the simulation size is 100000
    groups['Iterations'] = 100  # Assuming that 100 iterations were done
