"""
This will ask what HDF5 file you want loaded. The data will be stored in a nested dictionary.
The data is structured in groups, sub-groups and data-sets.

The group indicates the delta value
The sub-groups indicate the nu values
The data-sets contain the data (number of excited cells in grid for each time step).

groups are in the form: u'delta: 0.01' and the subgroups are in the form u'Nu: 0.13'
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt

print'\n'

kishans_nu = np.array([0.02,0.04,0.06,0.08,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,
                       0.29,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.1])
kishans_values = np.array([0.99981, 0.99983, 0.9998, 0.99968, 0.99772, 0.96099, 0.60984, 0.16381, 0.017807,
                           0.020737, 4.922e-05, 0.0001084, 0, 0, 0.99152, 0.86184, 0.29714, 0.039206, 0.0056277,
                           4.834e-05, 0.00082172, 0, 0, 9.406e-05, 0.99919])


def af_scan(data_set, size, pulse_rate):
    """
    Function will find where AF occurs in data-set. It will check whether or not the system is actually in AF or not
    by checking the length of the 'Normal heart beat' state.

    :param data_set:
    :param size:
    :param pulse_rate:
    :return:
    """

    # Assuming the System does not start in AF.
    raw_af = (data_set > 1.1 * size)
    neighbour_af = (raw_af[:-1] != raw_af[1:])
    neighbour_ind = np.where(neighbour_af == True)[0]  # Needs == even if the IDE says otherwise

    starting = neighbour_ind[1::2]
    ending = neighbour_ind[2::2]
    test_regions = np.array(zip(starting, ending))
    af_diff = np.diff(neighbour_ind)[1::2]
    filtered = (af_diff < 2 * pulse_rate)
    af_overwrite = test_regions[filtered]

    for region in af_overwrite:
        for loc in range(region[0], region[1]+1):
            raw_af[loc] = True

    return raw_af


def get_risk_data(delta_):
    """

    :param delta_:
    :return:
    """
    risk = []
    error_bars = []
    for nu_key in nu_range:
        risk.append(refined_data['delta: %s' % delta_]['nu: %s' % nu_key][0])
        error_bars.append(refined_data['delta: %s' % delta_]['nu: %s' % nu_key][1]/np.sqrt(iterations))

    return np.array(risk), np.array(error_bars)


def af_line_plot(delta_=None, nu_=None, iteration=None, normalised=False, scanned=False):
    """
    Creates a plot of the number of excited sites.
    :param delta_:
    :param nu_:
    :param iteration:
    :param normalised:
    :param scanned
    :return:
    """
    x = np.arange(len(raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu][iteration]))
    data_ = raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration]
    if normalised:
        data_ = raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration]/float(
            max(raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration]))
        label = 'Normalised number of excited cells'
    if scanned:
        data_ = af_scan(
            raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration], 200, np.array(para['Pulse Rate']))
        label = 'AF scan'
    plt.plot(x, data_, label=label)


def af_error_plot(delta_):
    """
    Creates error bar plots of different deltas
    :param delta_:
    :return:
    """
    y, err = get_risk_data(delta_)
    x = np.array(nu_range)

    plt.errorbar(x, y, yerr=err, fmt='o', label='delta: %s' % delta_)


file_name = raw_input("HDF5 to open: ")
h5data = h5py.File('%s.h5' % file_name, 'r')
h5par = h5py.File('%s_para.h5' % file_name, 'r')

print('List of items in the base directory:', h5data.items())
print('List of items in the base directory:', h5par.items())
print'\n'

# Displaying parameters on load
print "Parameters:"
para = h5par.get('parameters')
for i in para.iterkeys():
    print "%s: %s" % (i, np.array(para[i]))

delta_range = np.array(para.get('Delta'))
nu_range = np.array(para.get('Nu'))
sim_size = np.array(para.get('Simulation Size'))
iterations = np.array(para.get('Iterations'))
pr = np.array(para.get('Pulse Rate'))

print '\n'
raw_data = {}  # This is where all the data is stored

for i in h5data.iterkeys():  # Structuring the data into total_data
    grp = h5data.get(i)
    key1 = '%s' % i
    raw_data[key1] = {}
    for j in grp.iterkeys():
        s_grp = grp.get(j)
        key2 = '%s' % j
        temp_data = []
        for k in s_grp.iterkeys():
            data = np.array(s_grp.get(k))
            temp_data.append(data)

        raw_data[key1][key2] = temp_data

# Stores the data in tuples (mean, std)
refined_data = {}

for delta in delta_range:
    refined_data['delta: %s' % delta] = {}
    for nu in nu_range:
        risk_data = np.zeros(iterations)
        for i in range(iterations):
            risk_data[i] = np.mean(np.array(af_scan(raw_data[u'delta: %s' % delta][u'Nu: %s' % nu][i], 200, pr)))
        grouped_risk_data = (np.mean(risk_data), np.std(risk_data))

        refined_data['delta: %s' % delta]['nu: %s' % nu] = grouped_risk_data

plt.figure(1)
af_line_plot(0.05, 0.14, 1, normalised=True)
af_line_plot(0.05, 0.14, 1, normalised=False, scanned=True)
plt.hlines((200 * 1.1)/float(
            max(raw_data[u'delta: %s' % 0.05][u'Nu: %s' % 0.14][1])), 0, sim_size)

plt.figure(2)
#af_error_plot(0.01)
af_error_plot(0.05)
#af_error_plot(0.25)
plt.plot(kishans_nu, kishans_values, 'r^', label="kishan")
plt.grid(True)
plt.ylabel("Risk of AF")
plt.xlabel("Nu")
plt.legend()
plt.show()

plt.close()

h5data.close()
h5par.close()
