import numpy as np
import h5py

print'\n'

file_name = raw_input("HDF5 to open: ")
h5data = h5py.File('%s.h5' % file_name, 'r')
print('List of items in the base directory:', h5data.items())

total_data = {}

def af_scan(data, size, pulse_rate):

    # Assuming the System does not start in AF.
    raw_af = (data > 1.1 * 200)
    neighbour_af = (raw_af[:-1] != raw_af[1:])
    neighbour_ind = np.where(neighbour_af == True)[0]

    print len(neighbour_ind)
    print neighbour_ind
    starting = neighbour_ind[1::2]
    ending =  neighbour_ind[2::2]
    test_regions = np.array(zip(starting,ending))
    AF_diff = np.diff(neighbour_ind)[1::2]
    print test_regions
    print AF_diff
    print type(test_regions)
    
    """
    else:
        print "odd"
        starting = neighbour_ind[1:-1:2]
        ending = neighbour_ind[2::2]
        test_regions = np.array(zip(starting,ending))
        AF_diff = np.diff(neighbour_ind)[1::2]
        print type(test_regions)
    """

        # if (neighbour_ind[point+1] - index) < (2 * pulse_rate):
        #     print (index, neighbour_ind[index+1])


for i in h5data.iterkeys():
    group = h5data.get(i)
    key = '%s' % i
    temp_data = []
    for j in group.iterkeys():
        data = np.array(group.get(j))
        temp_data.append(data)

    total_data[key] = temp_data

af_scan(total_data[u'group0'][0], 200, 220)

print'\n'