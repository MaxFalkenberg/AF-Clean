import numpy as np
import h5py
import matplotlib.pyplot as plt
import theano.tensor as T
from theano import function
from theano.tensor.signal.conv import conv2d
from itertools import product
from numpy.fft import rfft
from numpy.fft import irfft


##############################################################################################################
"""
Functions used in Open.py
Option: Delta

Should change these so that they create one (more general).
"""


def af_scan(data_set, size, pulse_rate):
    """
    Function will find where AF occurs in data-set. It will check whether or not the system is actually in AF or not
    by checking the length of the 'Normal heart beat' state.

    :param data_set: Desired data set to scan.
    :param size: Size of the grid L.
    :param pulse_rate: Pulse Rate.
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
        for loc in range(region[0], region[1] + 1):
            raw_af[loc] = True

    return raw_af


def get_risk_data(delta_, nu_range, refined_data, iterations):
    """
    Function which gathers the risk data from refind_data and gives out the risk/errors for each delta.
    :param delta_: The delta value
    :param nu_range: The range of nu values
    :param refined_data: Empty list to append the data to
    :param iterations: Number of iterations of the data
    :return:
    """
    risk = []
    error_bars = []
    for nu_key in nu_range:
        risk.append(refined_data['delta: %s' % delta_]['nu: %s' % nu_key][0])
        error_bars.append(refined_data['delta: %s' % delta_]['nu: %s' % nu_key][1] / np.sqrt(iterations))

    return np.array(risk), np.array(error_bars)


def af_line_plot(raw_data, para, delta_=None, nu_=None, iteration=None, normalised=False, scanned=False):
    """
    Creates a plot of the number of excited sites.
    :param raw_data: The dictionary file containing all the data.
    :param para: The para subgroup from the parameter hdf5 file (might change to be more general)
    :param delta_: The Delta value
    :param nu_: The Nu value
    :param iteration: The Iteration that is plotted
    :param normalised: Normalises the data so that it lies between 0 and 1
    :param scanned: Scans the data for AF by using af_scan().
    :return:
    """
    x = np.arange(len(raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration]))
    data_ = raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration]
    label = None  # Just to assign it as something.
    if normalised:
        data_ = raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration] / float(
            max(raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration]))
        label = 'Normalised number of excited cells'
    if scanned:
        data_ = af_scan(
            raw_data[u'delta: %s' % delta_][u'Nu: %s' % nu_][iteration], 200, np.array(para['Pulse Rate']))
        label = 'AF scan'
    plt.plot(x, data_, label=label)


def af_error_plot(delta_, nu_range, refined_data, iterations):
    """
    Creates error bar plots of different deltas.
    :param delta_: Delta set you want to plot.
    :param nu_range: The nu_range to get plots for.
    :param refined_data: Dictionary of the refined data created from af_scan().
    :param iterations: Number of iterations of the data.
    :return:
    """
    y, err = get_risk_data(delta_, nu_range, refined_data, iterations)
    x = np.array(nu_range)
    plt.errorbar(x, y, yerr=err, fmt='o', label='delta: %s' % delta_)

##############################################################################################################
