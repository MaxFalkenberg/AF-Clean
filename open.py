"""
This will ask what HDF5 file you want loaded. The data will be stored in a nested dictionary.
The data is structured in groups, sub-groups and data-sets.

The group indicates the delta value
The sub-groups indicate the nu values
The data-sets contain the data (number of excited cells in grid for each time step).


"""

import numpy as np
import h5py

print'\n'

file_name = raw_input("HDF5 to open: ")
h5data = h5py.File('%s.h5' % file_name, 'r')
print('List of items in the base directory:', h5data.items())

def af_scan(data, size, pulse_rate):
    """
    Function will find where AF occurs in data-set. It will check whether or not the system is actually in AF or not
    by checking the length of the 'Normal heart beat' state.

    :param data:
    :param size:
    :param pulse_rate:
    :return:
    """

    # Assuming the System does not start in AF.
    raw_af = (data > 1.1 * size)
    neighbour_af = (raw_af[:-1] != raw_af[1:])
    neighbour_ind = np.where(neighbour_af == True)[0]

    starting = neighbour_ind[1::2]
    ending =  neighbour_ind[2::2]
    test_regions = np.array(zip(starting,ending))
    AF_diff = np.diff(neighbour_ind)[1::2]
    filter = (AF_diff < 2 * pulse_rate)
    AF_overwrite = test_regions[filter]

    for region in AF_overwrite:
        for i in range(region[0],region[1]+1):
            raw_af[i] = True

    return raw_af

total_data = {} #  This is where all the data is stored

for i in h5data.iterkeys(): #  Structuring the data into total_data
    grp = h5data.get(i)
    key1 = '%s' % i
    total_data[key1] = {}
    for j in grp.iterkeys():
        s_grp = grp.get(j)
        key2 = '%s' % j
        temp_data = []
        for k in s_grp.iterkeys():
            data = np.array(s_grp.get(k))
            temp_data.append(data)

    total_data[key1][key2] = temp_data

"""
groups are in the form: u'delta: 0.01' and the subgroups are in the form u'Nu: 0.13'
"""