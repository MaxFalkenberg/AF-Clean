import numpy as np
import h5py
import matplotlib.pyplot as plt
import theano.tensor as T
from theano import function
from theano.tensor.signal.conv import conv2d
from astropy.stats import LombScargle
from itertools import product
from numpy.fft import rfft
from numpy.fft import irfft
from scipy.stats import mode
from sklearn.tree import export_graphviz
import subprocess
import sys
import scipy.signal as ss
# import nolds
import scipy.stats as stats
import pandas as pd


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

"""
general convert function
"""


def sampling_convert(data, output, shape, rp, animation_grid):
    """
    :param data: Data to convert into animation format
    :param output: Output list containing data
    :param shape: Shape of grid
    :param rp: Refractory Period
    :param animation_grid: Animation grid that the animation is built on
    :return:
    """
    for index_data in data:
        animation_grid[(animation_grid > 0) & (animation_grid <= 50)] -= 1
        if index_data == []:  # could use <if not individual_data.any():> but this is more readable.
            current_state = animation_grid.copy()
            output.append(current_state)
        else:
            indices = np.unravel_index(index_data, shape)
            for ind in range(len(indices[0])):
                animation_grid[indices[0][ind]][indices[1][ind]] = rp
            current_state = animation_grid.copy()
            output.append(current_state)
    return output

##############################################################################################################

"""
Functions for rt_ecg.py animations
"""


def ani_convert(data, shape, rp, animation_grid):
    """
    Converts all the file data into arrays that can be animated.
    :param data: Data to convert into animation format.
    :param shape: shape of the animation grid.
    :param rp: Refractory period.
    :param animation_grid: The grid which the animation is built on.
    :return:
    """

    animation_grid[(animation_grid > 0) & (animation_grid <= rp)] -= 1
    if data == []:  # could use <if not individual_data.any():> but this is more readable.
        return animation_grid
    else:
        indices = np.unravel_index(data, shape)
        for ind in range(len(indices[0])):
            animation_grid[indices[0][ind]][indices[1][ind]] = rp
        return animation_grid


##############################################################################################################

"""
Function which involve the use of pandas dataframes
"""

def roll_dist(cp):
    """
    Returns the a grid of the distances between the critical points and ecg probes.
    :param cp:
    :return:
    """
    y, x = np.unravel_index(cp, (200, 200))
    pythag = np.zeros((200, 200), dtype='float')
    x_grid = np.copy(pythag)
    y_grid = np.copy(pythag)
    y_mid = float(len(y_grid) / 2)
    for i in range(len(pythag)):
        x_grid[:, i] = i
        y_grid[i] = i
    x_grid -= float(x)
    y_grid -= y_mid
    pythag += ((x_grid ** 2) + (y_grid ** 2)) ** 0.5
    return np.roll(pythag, int(y_mid + y), axis=0)


def feature_extract(number, ecg_vals, cp, probes):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.
    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    :param cp: The position of the critical point.
    :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    crit_point = cp.tolist() #Index of critical point
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    dist = []
    unit_vector_x = []
    unit_vector_y = []
    theta = []
    target = []
    cp = np.atleast_1d(cp)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    for i in range(len(cp)):
        def cp_vector(y_probe,x_probe):
            x_vector = int(x_probe) - x[i]
            y_vector = int(y_probe) - y[i]
            if y_vector > 100:
                y_vector -= 200
            elif y_vector <= -100:
                y_vector += 200

            r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
            c = (x_vector + (1j * y_vector)) /r
            theta = np.angle(c)
            return r,float(x_vector)/r,float(y_vector)/r,theta
        a,b,c,d = cp_vector(probes[number][0],probes[number][1])
        if a <= np.sqrt(200):
            t = 1
        else:
            t = 0
        dist.append(a)
        unit_vector_x.append(b)
        unit_vector_y.append(c)
        theta.append(d)
        target.append(t)
    nearest = [np.argmin(dist)]

    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-9:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ft.size, d=1.)
    freq_main = np.fft.rfftfreq(ft.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    period = int(1. / freq_main)
    ft2 = np.copy(ft)
    ft2[ft_max + 1:] = 0
    ift = irfft(ft2)
    start = np.argmax(ift[:(2*period)])
    #start = 0
    end = start + (2 * period)
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.

    ft_samp = rfft(sample_)  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(ft.size, d=1.)
    ft_samp_abs = np.absolute(ft_samp)  # Takes absolute value of FT
    ft_samp_max10 = np.argsort(ft_samp_abs)[-9:]  # Finds 9 largest frequency fundamentals

    grad = np.gradient(sample_)
    stat_points = []
    stat_diffs = []

    # FEATURE: Maximum value of sample ECG
    max_value = np.max(sample_)
    # FEATURE: Minimum value of sample ECG
    min_value = np.min(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    sample_int = np.sum(np.absolute(sample_))
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    sample_int_neg = np.sum(sample_[sample_ < 0.])
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)
    # FEATURE: Sum of all positive voltages
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # Feature: Sum of all negative voltages
    sample_int_neg = np.sum(sample_[sample_ < 0.])

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    grad_min = np.min(grad)
    # FEATURE: Difference of the above
    grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    grad_argdiff = grad_argmax - grad_argmin

    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            stat_points.append(i)
    # FEATURE: The number of stationary points
    n_stat_point = len(stat_points)
    # FEATURE: The position of the first stationary point
    arg_firststat = stat_points[0]

    """
    Think about a way to deal with nans in RFC (might not matter)
    """
    # for i in range(len(stat_points) - 1):
    #     try:
    #         stat_diffs.append(stat_points[i + 1] - stat_points[i])
    #         if len(stat_diffs) < 6:
    #             np.pad(stat_diffs, (0, 6 - len(stat_diffs)), 'constant')
    #     except:
    #         break



    # FEATURE: Largest 9 frequencies in sample ECG. Largest first.

    largest_ft_freq = freq_samp[ft_samp_max10[::-1]].tolist()
    # FEATURE: Absolute values of largest 9 freqs
    largest_ft_mag = ft_samp_abs[ft_samp_max10[::-1]].tolist()
    # FEATURE: Sum of absolute values
    largest_sum = np.sum(ft_samp_abs[ft_samp_max10[::-1]])
    # FEATURE: Absolute values normalised by sum.
    largest_ft_rel_mag = [mag/largest_sum for mag in largest_ft_mag]

    features = np.array([max_value, min_value, minmax_dif, sample_int, sample_len, grad_max, grad_min, grad_diff,
                         grad_argmax, grad_argmin, grad_argdiff, n_stat_point, arg_firststat]
                        + largest_ft_freq + largest_ft_mag + largest_ft_rel_mag +
                        [largest_sum] + cp.tolist() + [probe_point] + dist + unit_vector_x + unit_vector_y + theta + target + nearest + [start] + [end])
    return features


def feature_extract2(number, ecg_vals, cp, probes):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.
    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    :param cp: The position of the critical point.
    :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    crit_point = cp.tolist() #Index of critical point
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    dist = []
    unit_vector_x = []
    unit_vector_y = []
    theta = []
    target = []
    cp = np.atleast_1d(cp)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    for i in range(len(cp)):
        def cp_vector(y_probe,x_probe):
            x_vector = int(x_probe) - x[i]
            y_vector = int(y_probe) - y[i]
            if y_vector > 100:
                y_vector -= 200
            elif y_vector <= -100:
                y_vector += 200

            r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
            c = (x_vector + (1j * y_vector)) /r
            theta = np.angle(c)
            return r,float(x_vector)/r,float(y_vector)/r,theta
        a,b,c,d = cp_vector(probes[number][0],probes[number][1])
        if a <= np.sqrt(200):
            t = 1
        else:
            t = 0
        dist.append(a)
        unit_vector_x.append(b)
        unit_vector_y.append(c)
        theta.append(d)
        target.append(t)
    nearest = [np.argmin(dist)]

    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-4:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ecg.size, d=1.)
    freq_main = np.fft.rfftfreq(ecg.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ft[:ft_max] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:period])
    #start = 0
    end = start + period
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.

    ft_samp = rfft(sample_)  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(sample_.size, d=1.)
    ft_samp_abs = np.absolute(ft_samp)  # Takes absolute value of FT
    ft_samp_abs_rel1 = ft_samp_abs / ft_samp_abs[1]
    ft_samp_abs_rel2 = ft_samp_abs / ft_samp_abs[2]
    ft_samp_abs_rel3 = ft_samp_abs / ft_samp_abs[3]
    # ft_samp_max10 = np.argsort(ft_samp_abs)[-9:]  # Finds 9 largest frequency fundamentals

    grad = np.gradient(sample_)
    stat_points = []
    stat_diffs = []

    # FEATURE: Maximum value of sample ECG
    max_value = np.max(sample_)
    max_arg = np.argmax(sample_)
    # FEATURE: Minimum value of sample ECG
    min_value = np.min(sample_)
    min_arg = np.argmin(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    minmax_half = (max_value + min_value)/2
    try:
        arghalf = np.argwhere(sample_[max_arg:min_arg] < minmax_half)[0]
    except:
        arghalf = np.array([0])
    half_ratio = float(arghalf - max_arg) / float(min_arg - max_arg)
    std_full = np.std(sample_)
    std_premax = np.std(sample_[:max_arg])
    try:
        std_minmax = np.std(sample_[max_arg:min_arg])
    except:
        std_minmax = 0
    std_postmin = np.std(sample_[min_arg:])
    n_extrema_max = len(ss.argrelextrema(sample_,np.greater))
    n_extrema_min = len(ss.argrelextrema(sample_,np.less))
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    sample_int = np.sum(np.absolute(sample_))
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    sample_int_neg = np.sum(sample_[sample_ < 0.])
    sample_int_ratio = float(sample_int_pos) / float(sample_int_neg)
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)
    # FEATURE: Sum of all positive voltages
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # Feature: Sum of all negative voltages
    sample_int_neg = np.sum(sample_[sample_ < 0.])

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    grad_min = np.min(grad)
    # FEATURE: Difference of the above
    grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    grad_argdiff = grad_argmax - grad_argmin
    grad_minmax_mean = np.mean(grad[max_arg:min_arg + 1])

    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            stat_points.append(i)
    # FEATURE: The number of stationary points
    n_stat_point = len(stat_points)
    # FEATURE: The position of the first stationary point
    arg_firststat = stat_points[0]

    """
    Think about a way to deal with nans in RFC (might not matter)
    """
    # for i in range(len(stat_points) - 1):
    #     try:
    #         stat_diffs.append(stat_points[i + 1] - stat_points[i])
    #         if len(stat_diffs) < 6:
    #             np.pad(stat_diffs, (0, 6 - len(stat_diffs)), 'constant')
    #     except:
    #         break



    # FEATURE: Largest 9 frequencies in sample ECG. Largest first.

    # largest_ft_freq = freq_samp[ft_samp_max10[::-1]].tolist()
    # FEATURE: Absolute values of largest 9 freqs
    # largest_ft_mag = ft_samp_abs[ft_samp_max10[::-1]].tolist()
    # FEATURE: Sum of absolute values
    # largest_sum = np.sum(ft_samp_abs[ft_samp_max10[::-1]])
    # FEATURE: Absolute values normalised by sum.
    # largest_ft_rel_mag = [mag/largest_sum for mag in largest_ft_mag]
    # if len(freq_samp) != 31 or len(ft_samp_abs) != 31 or len(ft_samp_abs_rel) != 31:
    #     print(len(freq_samp),len(ft_samp_abs),len(ft_samp_abs_rel))
    features = np.array([max_value, min_value, minmax_dif, max_arg,min_arg,minmax_half,arghalf[0],half_ratio,
                        std_full,std_premax,std_minmax,std_postmin,n_extrema_max,n_extrema_min,sample_int_pos,
                        sample_int_neg,sample_int_ratio,grad_minmax_mean,
                        sample_int, sample_len, grad_max, grad_min, grad_diff,
                         grad_argmax, grad_argmin, grad_argdiff, n_stat_point, arg_firststat]
                        + freq_samp.tolist() + ft_samp_abs.tolist() + ft_samp_abs_rel1.tolist() + ft_samp_abs_rel2.tolist() + ft_samp_abs_rel3.tolist()
                        + cp.tolist() + [probe_point] + dist + unit_vector_x + unit_vector_y + theta + target + nearest + [start] + [end])
    return features

def feature_extract3(index, number, ecg_vals, cp, probes):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.
    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    :param cp: The position of the critical point.
    :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    crit_point = cp.tolist() #Index of critical point
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    dist = []
    unit_vector_x = []
    unit_vector_y = []
    vec_x = []
    vec_y = []
    theta = []
    target = []
    cp = np.atleast_1d(cp)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    for i in range(len(cp)):
        def cp_vector(y_probe,x_probe):
            x_vector = int(x_probe) - x[i]
            y_vector = int(y_probe) - y[i]
            if y_vector > 100:
                y_vector -= 200
            elif y_vector <= -100:
                y_vector += 200

            r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
            c = (x_vector + (1j * y_vector)) /r
            theta = np.angle(c)
            return r,float(x_vector)/r,float(y_vector)/r,theta,float(x_vector),float(y_vector)
        a,b,c,d,e,f = cp_vector(probes[number][0],probes[number][1])
        if a <= np.sqrt(200):
            t = 1
        else:
            t = 0
        dist.append(a)
        unit_vector_x.append(b)
        unit_vector_y.append(c)
        theta.append(d)
        target.append(t)
        vec_x.append(e)
        vec_y.append(f)
    nearest = [np.argmin(dist)]

    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-4:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ecg.size, d=1.)
    freq_main = np.fft.rfftfreq(ecg.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ft[:ft_max] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:period])
    #start = 0
    end = start + period
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.
    sample_double =ecg[start:end + (2 * period)]
    length,minmax,mean,var,skew,kurt = stats.describe(sample_)
    min_value,max_value = minmax
    #mean = np.mean(sample_)
    ft_samp = rfft(sample_)[:4]  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(sample_.size, d=1.)[:4]
    ft_samp_abs = np.absolute(ft_samp)  # Takes absolute value of FT
    ft_samp_abs_rel1 = ft_samp_abs / ft_samp_abs[1]
    ft_samp_abs_rel2 = ft_samp_abs / ft_samp_abs[2]
    ft_samp_abs_rel3 = ft_samp_abs / ft_samp_abs[3]
    # ft_samp_max10 = np.argsort(ft_samp_abs)[-9:]  # Finds 9 largest frequency fundamentals

    grad = np.gradient(sample_)
    stat_points = []
    stat_diffs = []

    #entropy = nolds.sampen(sample_double)
    #hurst = nolds.hurst_rs(sample_double)
    # dfa = nolds.dfa(sample_double)
    #corr_dim = nolds.corr_dim(sample_double,1)

    # FEATURE: Maximum value of sample ECG
    #max_value = np.max(sample_)
    max_arg = np.argmax(sample_)
    # FEATURE: Minimum value of sample ECG
    #min_value = np.min(sample_)
    min_arg = np.argmin(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    minmax_half = (max_value + min_value)/2
    try:
        arghalf = np.argwhere(sample_[max_arg:min_arg] < minmax_half)[0]
    except:
        arghalf = np.array([0])
    half_ratio = float(arghalf - max_arg) / float(min_arg - max_arg)
    #std_full = np.std(sample_)

    std_postmin = np.std(sample_[min_arg:])
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    sample_int = np.sum(np.absolute(sample_))
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    sample_int_neg = np.sum(sample_[sample_ < 0.])
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)
    # FEATURE: Sum of all positive voltages
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # Feature: Sum of all negative voltages
    sample_int_neg = np.sum(sample_[sample_ < 0.])

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    grad_min = np.min(grad)
    # FEATURE: Difference of the above
    grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    grad_argdiff = grad_argmax - grad_argmin


    g_temp = grad[max_arg:min_arg + 1]
    if len(g_temp) == 0:
        g_temp = grad[min_arg:max_arg + 1]
        grad_minmax_mean =  - np.mean(g_temp)
    else:
        grad_minmax_mean = np.mean(g_temp)

    if len(sample_[:max_arg]) == 0:
        std_premax = - np.std(sample_[max_arg:])
    else:
        std_premax = np.std(sample_[:max_arg])
    if len(sample_[max_arg:min_arg]) == 0:
        std_minmax =  - np.std(sample_[min_arg:max_arg])
    else:
        std_minmax = np.std(sample_[max_arg:min_arg])


    covariance = np.cov(sample_)
    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            stat_points.append(i)
    # FEATURE: The position of the first stationary point
    arg_firststat = stat_points[0]

    """
    Think about a way to deal with nans in RFC (might not matter)
    """
    #entropy,hurst,corr_dim,dfa,
    features = np.array([start,index,number,covariance,mean,skew,kurt,max_value, min_value, minmax_dif, max_arg,min_arg,minmax_half,arghalf[0],half_ratio,
                        var,std_premax,std_minmax,std_postmin,sample_int_pos,
                        sample_int_neg,grad_minmax_mean,
                        sample_int, sample_len, grad_max, grad_min, grad_diff,
                         grad_argmax, grad_argmin, grad_argdiff, arg_firststat]
                        + freq_samp.tolist() + ft_samp_abs.tolist() + ft_samp_abs_rel1.tolist() + ft_samp_abs_rel2.tolist() + ft_samp_abs_rel3.tolist()
                        + cp.tolist() + [probe_point] + dist + vec_x + vec_y + unit_vector_x + unit_vector_y + theta + target + nearest)
    return features

def feature_extract_nu(number, ecg_vals, cp, probes, nu):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.
    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    :param cp: The position of the critical point.
    :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    crit_point = cp.tolist() #Index of critical point
    nu = nu.tolist()
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    dist = []
    unit_vector_x = []
    unit_vector_y = []
    vec_x = []
    vec_y = []
    theta = []
    target = []
    cp = np.atleast_1d(cp)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    for i in range(len(cp)):
        def cp_vector(y_probe,x_probe):
            x_vector = int(x_probe) - x[i]
            y_vector = int(y_probe) - y[i]
            if y_vector > 100:
                y_vector -= 200
            elif y_vector <= -100:
                y_vector += 200

            r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
            c = (x_vector + (1j * y_vector)) /r
            theta = np.angle(c)
            return r,float(x_vector)/r,float(y_vector)/r,theta,float(x_vector),float(y_vector)
        a,b,c,d,e,f = cp_vector(probes[number][0],probes[number][1])
        if a <= np.sqrt(200):
            t = 1
        else:
            t = 0
        dist.append(a)
        unit_vector_x.append(b)
        unit_vector_y.append(c)
        theta.append(d)
        target.append(t)
        vec_x.append(e)
        vec_y.append(f)
    nearest = [np.argmin(dist)]

    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-4:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ecg.size, d=1.)
    freq_main = np.fft.rfftfreq(ecg.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ft[:ft_max] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:period])
    #start = 0
    end = start + period
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.
    sample_double =ecg[start:end + (2 * period)]
    length,minmax,mean,var,skew,kurt = stats.describe(sample_)
    min_value,max_value = minmax
    #mean = np.mean(sample_)
    ft_samp = rfft(sample_)[:4]  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(sample_.size, d=1.)[:4]
    ft_samp_abs = np.absolute(ft_samp)  # Takes absolute value of FT
    ft_samp_abs_rel1 = ft_samp_abs / ft_samp_abs[1]
    ft_samp_abs_rel2 = ft_samp_abs / ft_samp_abs[2]
    ft_samp_abs_rel3 = ft_samp_abs / ft_samp_abs[3]
    # ft_samp_max10 = np.argsort(ft_samp_abs)[-9:]  # Finds 9 largest frequency fundamentals

    grad = np.gradient(sample_)
    stat_points = []
    stat_diffs = []

    #entropy = nolds.sampen(sample_double)
    #hurst = nolds.hurst_rs(sample_double)
    # dfa = nolds.dfa(sample_double)
    #corr_dim = nolds.corr_dim(sample_double,1)

    # FEATURE: Maximum value of sample ECG
    #max_value = np.max(sample_)
    max_arg = np.argmax(sample_)
    # FEATURE: Minimum value of sample ECG
    #min_value = np.min(sample_)
    min_arg = np.argmin(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    minmax_half = (max_value + min_value)/2
    try:
        arghalf = np.argwhere(sample_[max_arg:min_arg] < minmax_half)[0]
    except:
        arghalf = np.array([0])
    half_ratio = float(arghalf - max_arg) / float(min_arg - max_arg)
    #std_full = np.std(sample_)

    std_postmin = np.std(sample_[min_arg:])
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    sample_int = np.sum(np.absolute(sample_))
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    sample_int_neg = np.sum(sample_[sample_ < 0.])
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)
    # FEATURE: Sum of all positive voltages
    sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # Feature: Sum of all negative voltages
    sample_int_neg = np.sum(sample_[sample_ < 0.])

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    grad_min = np.min(grad)
    # FEATURE: Difference of the above
    grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    grad_argdiff = grad_argmax - grad_argmin


    g_temp = grad[max_arg:min_arg + 1]
    if len(g_temp) == 0:
        g_temp = grad[min_arg:max_arg + 1]
        grad_minmax_mean =  - np.mean(g_temp)
    else:
        grad_minmax_mean = np.mean(g_temp)

    if len(sample_[:max_arg]) == 0:
        std_premax = - np.std(sample_[max_arg:])
    else:
        std_premax = np.std(sample_[:max_arg])
    if len(sample_[max_arg:min_arg]) == 0:
        std_minmax =  - np.std(sample_[min_arg:max_arg])
    else:
        std_minmax = np.std(sample_[max_arg:min_arg])


    covariance = np.cov(sample_)
    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            stat_points.append(i)
    # FEATURE: The position of the first stationary point
    arg_firststat = stat_points[0]

    """
    Think about a way to deal with nans in RFC (might not matter)
    """
    #entropy,hurst,corr_dim,dfa,
    features = np.array([covariance,mean,skew,kurt,max_value, min_value, minmax_dif, max_arg,min_arg,minmax_half,arghalf[0],half_ratio,
                        var,std_premax,std_minmax,std_postmin,sample_int_pos,
                        sample_int_neg,grad_minmax_mean,
                        sample_int, sample_len, grad_max, grad_min, grad_diff,
                         grad_argmax, grad_argmin, grad_argdiff, arg_firststat]
                        + freq_samp.tolist() + ft_samp_abs.tolist() + ft_samp_abs_rel1.tolist() + ft_samp_abs_rel2.tolist() + ft_samp_abs_rel3.tolist()
                        + cp.tolist() + [probe_point] + dist + vec_x + vec_y + unit_vector_x + unit_vector_y + theta + target + nearest + [nu])
    return features

def feature_extract_multi(number, ecg_vals, cp, probes, nu):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.
    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    :param cp: The position of the critical point.
    :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    index = np.arange(1,len(ecg) + 1)
    f,p = LombScargle(index,ecg).autopower()
    per = int(np.round(1./f[np.argmax(p[(1./f) > 5.])]))
    crit_point = cp.tolist() #Index of critical point
    nu = nu.tolist()
    # probe_number = probe_number.tolist()
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    dist = []
    unit_vector_x = []
    unit_vector_y = []
    vec_x = []
    vec_y = []
    theta = []
    target = []
    multi_target = []
    cp = np.atleast_1d(cp)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    for i in range(len(cp)):
        def cp_vector(y_probe,x_probe):
            x_vector = int(x_probe) - x[i]
            y_vector = int(y_probe) - y[i]
            if y_vector > 100:
                y_vector -= 200
            elif y_vector <= -100:
                y_vector += 200

            r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
            c = (x_vector + (1j * y_vector)) /r
            theta = np.angle(c)
            return r,float(x_vector)/r,float(y_vector)/r,theta,float(x_vector),float(y_vector)
        a,b,c,d,e,f = cp_vector(probes[number][0],probes[number][1])
        if a <= np.sqrt(200):
            t = 1
        else:
            t = 0
        if np.absolute(e) < 4 and np.absolute(f) < 4:
            m = True
        else:
            m = False
        dist.append(a)
        unit_vector_x.append(b)
        unit_vector_y.append(c)
        theta.append(d)
        target.append(t)
        vec_x.append(e)
        vec_y.append(f)
        multi_target.append(m)
    nearest = [np.argmin(dist)]

    ft = rfft(ecg[:2*per])  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-3:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ecg.size, d=1.)
    freq_main = np.fft.rfftfreq(ecg.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ft[:ft_max] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:per])
    #start = 0
    end = start + period
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.
    # sample_double =ecg[start:end + (2 * period)]
    length,minmax,mean,var,skew,kurt = stats.describe(sample_)
    min_value,max_value = minmax
    #mean = np.mean(sample_)
    ft_samp = rfft(sample_)[:3]  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(sample_.size, d=1.)[:9]
    ft_samp_abs = np.absolute(ft_samp)  # Takes absolute value of FT
    # print freq_samp, end - start
    ft_samp_abs_rel2 = ft_samp_abs / ft_samp_abs[2]
    # ft_samp_max10 = np.argsort(ft_samp_abs)[-9:]  # Finds 9 largest frequency fundamentals

    grad = np.gradient(sample_)
    stat_points = []
    # stat_diffs = []

    #entropy = nolds.sampen(sample_double)
    #hurst = nolds.hurst_rs(sample_double)
    # dfa = nolds.dfa(sample_double)
    #corr_dim = nolds.corr_dim(sample_double,1)

    # FEATURE: Maximum value of sample ECG
    #max_value = np.max(sample_)
    max_arg = np.argmax(sample_)
    # FEATURE: Minimum value of sample ECG
    #min_value = np.min(sample_)
    min_arg = np.argmin(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    minmax_half = (max_value + min_value)/2
    try:
        arghalf = np.argwhere(sample_[max_arg:min_arg] < minmax_half)[0]
    except:
        arghalf = np.array([0])
    #half_ratio = float(arghalf - max_arg) / float(min_arg - max_arg)
    #std_full = np.std(sample_)

    std_postmin = np.std(sample_[min_arg:])
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    # sample_int = np.sum(np.absolute(sample_))
    # sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # sample_int_neg = np.sum(sample_[sample_ < 0.])
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)
    # FEATURE: Sum of all positive voltages
    # sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # Feature: Sum of all negative voltages
    # sample_int_neg = np.sum(sample_[sample_ < 0.])

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    # grad_min = np.min(grad)
    # FEATURE: Difference of the above
    # grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    # grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    # grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    # grad_argdiff = grad_argmax - grad_argmin


    # g_temp = grad[max_arg:min_arg + 1]
    # if len(g_temp) == 0:
    #     g_temp = grad[min_arg:max_arg + 1]
    #     grad_minmax_mean =  - np.mean(g_temp)
    # else:
    #     grad_minmax_mean = np.mean(g_temp)
    #
    # if len(sample_[:max_arg]) == 0:
    #     std_premax = - np.std(sample_[max_arg:])
    # else:
    #     std_premax = np.std(sample_[:max_arg])
    # if len(sample_[max_arg:min_arg]) == 0:
    #     std_minmax =  - np.std(sample_[min_arg:max_arg])
    # else:
    #     std_minmax = np.std(sample_[max_arg:min_arg])


    # covariance = np.cov(sample_)
    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            stat_points.append(i)
    # FEATURE: The position of the first stationary point
    arg_firststat = stat_points[0]

    """
    Think about a way to deal with nans in RFC (might not matter)
    """
    #entropy,hurst,corr_dim,dfa,
    features = np.array([start, mean,skew,kurt,max_value, min_value, minmax_dif, max_arg,min_arg,minmax_half,arghalf[0],
                        std_postmin,
                        sample_len, grad_max,
                         arg_firststat]
                        + ft_samp_abs.tolist() +  ft_samp_abs_rel2.tolist()
                        + cp.tolist() + [probe_point] + dist + vec_x + vec_y + unit_vector_x + unit_vector_y + theta + target + multi_target + nearest + [nu]+ [int(number)/9])
    return features


def feature_extract_multi_test(number, ecg_vals, cp, probes, nu):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.
    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    :param cp: The position of the critical point.
    :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    index = np.arange(1,len(ecg) + 1)
    # f,p = LombScargle(index,ecg).autopower()
    # per = int(np.round(1./f[np.argmax(p[(1./f) > 5.])]))
    crit_point = cp.tolist() #Index of critical point
    nu = nu.tolist()
    # probe_number = probe_number.tolist()
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    dist = []
    unit_vector_x = []
    unit_vector_y = []
    vec_x = []
    vec_y = []
    theta = []
    target = []
    multi_target = []
    cp = np.atleast_1d(cp)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    for i in range(len(cp)):
        def cp_vector(y_probe,x_probe):
            x_vector = int(x_probe) - x[i]
            y_vector = int(y_probe) - y[i]
            if y_vector > 100:
                y_vector -= 200
            elif y_vector <= -100:
                y_vector += 200

            r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
            c = (x_vector + (1j * y_vector)) /r
            theta = np.angle(c)
            return r,float(x_vector)/r,float(y_vector)/r,theta,float(x_vector),float(y_vector)
        a,b,c,d,e,f = cp_vector(probes[number][0],probes[number][1])
        if a <= np.sqrt(200):
            t = 1
        else:
            t = 0
        if np.absolute(e) < 4 and np.absolute(f) < 4:
            m = True
        else:
            m = False
        dist.append(a)
        unit_vector_x.append(b)
        unit_vector_y.append(c)
        theta.append(d)
        target.append(t)
        vec_x.append(e)
        vec_y.append(f)
        multi_target.append(m)
    nearest = [np.argmin(dist)]

    signs = np.sign(np.diff(np.sign(ecg)))
    crossovers = np.argwhere(signs == -1).flatten()
    # print crossovers
    start = crossovers[0]
    noise = []
    for i in range(1,len(crossovers)):
        per = crossovers[i] - start
        if per >= 50:
            end = crossovers[i]
            break
        else:
            noise.append(crossovers[i])
    ecg = ecg[:2*per]
    # print start, end
    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-3:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ecg.size, d=1.)
    freq_main = np.fft.rfftfreq(ecg.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ft[:ft_max] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:period])
    #start = 0
    end = start + period
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.
    # sample_double =ecg[start:end + (2 * period)]
    length,minmax,mean,var,skew,kurt = stats.describe(sample_)
    min_value,max_value = minmax
    #mean = np.mean(sample_)
    ft_samp = rfft(sample_)[:3]  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(sample_.size, d=1.)[:9]
    ft_samp_abs = np.absolute(ft_samp)  # Takes absolute value of FT
    # print freq_samp, end - start
    # print len(sample_),start,per, np.gradient(np.sign(ecg))
    ft_samp_abs_rel2 = ft_samp_abs / ft_samp_abs[2]
    # ft_samp_max10 = np.argsort(ft_samp_abs)[-9:]  # Finds 9 largest frequency fundamentals

    grad = np.gradient(sample_)
    stat_points = []
    # stat_diffs = []

    #entropy = nolds.sampen(sample_double)
    #hurst = nolds.hurst_rs(sample_double)
    # dfa = nolds.dfa(sample_double)
    #corr_dim = nolds.corr_dim(sample_double,1)

    # FEATURE: Maximum value of sample ECG
    #max_value = np.max(sample_)
    max_arg = np.argmax(sample_)
    # FEATURE: Minimum value of sample ECG
    #min_value = np.min(sample_)
    min_arg = np.argmin(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    minmax_half = (max_value + min_value)/2
    try:
        arghalf = np.argwhere(sample_[max_arg:min_arg] < minmax_half)[0]
    except:
        arghalf = np.array([0])
    #half_ratio = float(arghalf - max_arg) / float(min_arg - max_arg)
    #std_full = np.std(sample_)

    std_postmin = np.std(sample_[min_arg:])
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    # sample_int = np.sum(np.absolute(sample_))
    # sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # sample_int_neg = np.sum(sample_[sample_ < 0.])
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)
    # FEATURE: Sum of all positive voltages
    # sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # Feature: Sum of all negative voltages
    # sample_int_neg = np.sum(sample_[sample_ < 0.])

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    # grad_min = np.min(grad)
    # FEATURE: Difference of the above
    # grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    # grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    # grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    # grad_argdiff = grad_argmax - grad_argmin


    # g_temp = grad[max_arg:min_arg + 1]
    # if len(g_temp) == 0:
    #     g_temp = grad[min_arg:max_arg + 1]
    #     grad_minmax_mean =  - np.mean(g_temp)
    # else:
    #     grad_minmax_mean = np.mean(g_temp)
    #
    # if len(sample_[:max_arg]) == 0:
    #     std_premax = - np.std(sample_[max_arg:])
    # else:
    #     std_premax = np.std(sample_[:max_arg])
    # if len(sample_[max_arg:min_arg]) == 0:
    #     std_minmax =  - np.std(sample_[min_arg:max_arg])
    # else:
    #     std_minmax = np.std(sample_[max_arg:min_arg])


    # covariance = np.cov(sample_)
    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            stat_points.append(i)
    # FEATURE: The position of the first stationary point
    arg_firststat = stat_points[0]

    """
    Think about a way to deal with nans in RFC (might not matter)
    """
    #entropy,hurst,corr_dim,dfa,
    features = np.array([start, mean,skew,kurt,max_value, min_value, minmax_dif, max_arg,min_arg,minmax_half,arghalf[0],
                        std_postmin,
                        sample_len, grad_max,
                         arg_firststat]
                        + ft_samp_abs.tolist() +  ft_samp_abs_rel2.tolist()
                        + cp.tolist() + [probe_point] + dist + vec_x + vec_y + unit_vector_x + unit_vector_y + theta + target + multi_target + nearest + [nu]+ [int(number)/9])
    return features


def feature_extract_multi_test_rt(number, ecg_vals):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.

    USING THIS FOR FEATURE EXTRACTION IN REAL TIME (ONLY RETURNS LIST WITHOUT RELATIVE POSITION IDENTIFIERS)

    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    # :param cp: The position of the critical point.
    # :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    index = np.arange(1,len(ecg) + 1)
    # f,p = LombScargle(index,ecg).autopower()
    # per = int(np.round(1./f[np.argmax(p[(1./f) > 5.])]))

    # DONT THINK I NEED ANY OF THIS (JUST POSITION INFO)

    # crit_point = cp.tolist() #Index of critical point
    # # probe_number = probe_number.tolist()
    # probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    # y,x = np.unravel_index(cp,(200,200))
    # # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP
    #
    # dist = []
    # unit_vector_x = []
    # unit_vector_y = []
    # vec_x = []
    # vec_y = []
    # theta = []
    # target = []
    # multi_target = []
    # cp = np.atleast_1d(cp)
    # x = np.atleast_1d(x)
    # y = np.atleast_1d(y)
    # for i in range(len(cp)):
    #     def cp_vector(y_probe,x_probe):
    #         x_vector = int(x_probe) - x[i]
    #         y_vector = int(y_probe) - y[i]
    #         if y_vector > 100:
    #             y_vector -= 200
    #         elif y_vector <= -100:
    #             y_vector += 200
    #
    #         r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
    #         c = (x_vector + (1j * y_vector)) /r
    #         theta = np.angle(c)
    #         return r,float(x_vector)/r,float(y_vector)/r,theta,float(x_vector),float(y_vector)
    #     a,b,c,d,e,f = cp_vector(probes[number][0],probes[number][1])
    #     if a <= np.sqrt(200):
    #         t = 1
    #     else:
    #         t = 0
    #     if np.absolute(e) < 4 and np.absolute(f) < 4:
    #         m = True
    #     else:
    #         m = False
    #     dist.append(a)
    #     unit_vector_x.append(b)
    #     unit_vector_y.append(c)
    #     theta.append(d)
    #     target.append(t)
    #     vec_x.append(e)
    #     vec_y.append(f)
    #     multi_target.append(m)
    # nearest = [np.argmin(dist)]

    signs = np.sign(np.diff(np.sign(ecg)))
    crossovers = np.argwhere(signs == -1).flatten()
    # print crossovers
    start = crossovers[0]
    noise = []
    for i in range(1,len(crossovers)):
        per = crossovers[i] - start
        if per >= 50:
            end = crossovers[i]
            break
        else:
            noise.append(crossovers[i])
    # print end - start
    ecg = ecg[:2*per]
    # print start, end
    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-3:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ecg.size, d=1.)
    freq_main = np.fft.rfftfreq(ecg.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    # period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ft[:ft_max] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:per])
    #start = 0
    end = start + per
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.
    # sample_double =ecg[start:end + (2 * period)]
    length,minmax,mean,var,skew,kurt = stats.describe(sample_)
    min_value,max_value = minmax
    #mean = np.mean(sample_)
    ft_samp = rfft(sample_)[:3]  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(sample_.size, d=1.)[:9]
    ft_samp_abs = np.absolute(ft_samp)  # Takes absolute value of FT
    # print freq_samp, end - start
    # print len(sample_),start,per, np.gradient(np.sign(ecg))
    ft_samp_abs_rel2 = ft_samp_abs / ft_samp_abs[2]
    # ft_samp_max10 = np.argsort(ft_samp_abs)[-9:]  # Finds 9 largest frequency fundamentals

    grad = np.gradient(sample_)
    stat_points = []
    # stat_diffs = []

    #entropy = nolds.sampen(sample_double)
    #hurst = nolds.hurst_rs(sample_double)
    # dfa = nolds.dfa(sample_double)
    #corr_dim = nolds.corr_dim(sample_double,1)

    # FEATURE: Maximum value of sample ECG
    #max_value = np.max(sample_)
    max_arg = np.argmax(sample_)
    # FEATURE: Minimum value of sample ECG
    #min_value = np.min(sample_)
    min_arg = np.argmin(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    minmax_half = (max_value + min_value)/2
    try:
        arghalf = np.argwhere(sample_[max_arg:min_arg] < minmax_half)[0]
    except:
        arghalf = np.array([0])
    #half_ratio = float(arghalf - max_arg) / float(min_arg - max_arg)
    #std_full = np.std(sample_)

    std_postmin = np.std(sample_[min_arg:])
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    # sample_int = np.sum(np.absolute(sample_))
    # sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # sample_int_neg = np.sum(sample_[sample_ < 0.])
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)
    # FEATURE: Sum of all positive voltages
    # sample_int_pos = np.sum(sample_[sample_ >= 0.])
    # Feature: Sum of all negative voltages
    # sample_int_neg = np.sum(sample_[sample_ < 0.])

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    # grad_min = np.min(grad)
    # FEATURE: Difference of the above
    # grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    # grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    # grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    # grad_argdiff = grad_argmax - grad_argmin


    # g_temp = grad[max_arg:min_arg + 1]
    # if len(g_temp) == 0:
    #     g_temp = grad[min_arg:max_arg + 1]
    #     grad_minmax_mean =  - np.mean(g_temp)
    # else:
    #     grad_minmax_mean = np.mean(g_temp)
    #
    # if len(sample_[:max_arg]) == 0:
    #     std_premax = - np.std(sample_[max_arg:])
    # else:
    #     std_premax = np.std(sample_[:max_arg])
    # if len(sample_[max_arg:min_arg]) == 0:
    #     std_minmax =  - np.std(sample_[min_arg:max_arg])
    # else:
    #     std_minmax = np.std(sample_[max_arg:min_arg])


    # covariance = np.cov(sample_)
    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            stat_points.append(i)
    # FEATURE: The position of the first stationary point
    arg_firststat = stat_points[0]

    """
    Think about a way to deal with nans in RFC (might not matter)
    """
    #entropy,hurst,corr_dim,dfa,
    features = np.array([start, mean,skew,kurt,max_value, min_value, minmax_dif, max_arg,min_arg,minmax_half,arghalf[0],
                        std_postmin,
                        sample_len, grad_max,
                         arg_firststat]
                        + ft_samp_abs.tolist() +  ft_samp_abs_rel2.tolist())
                        # RElative position identifiers.
                        # + cp.tolist() + [probe_point] + dist + vec_x + vec_y + unit_vector_x + unit_vector_y + theta + target + multi_target + nearest + [nu]+ [int(number)/9])
    return features


def feature_extract_keras(number, ecg_vals, cp, probes, nu, min = None):
    """
    Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format.
    :param number: Index in data.
    :param ecg_vals: The ecg voltages.
    :param cp: The position of the critical point.
    :param probes: The probe position.
    :return:
    """
    ecg = ecg_vals[number]
    index = np.arange(1,len(ecg) + 1)
    crit_point = cp.tolist() #Index of critical point
    nu = nu.tolist()
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    vec_x = []
    vec_y = []


    def cp_vector(y_probe,x_probe):
        x_vector = int(x_probe) - x
        y_vector = int(y_probe) - y
        if y_vector > 100:
            y_vector -= 200
        elif y_vector <= -100:
            y_vector += 200

        return float(x_vector),float(y_vector)
    x,y = cp_vector(probes[number][0],probes[number][1])
    x += 200.
    y += 200.
    x /= 400.
    y /= 400.

    vec_x.append(x)
    vec_y.append(y)

    signs = np.sign(np.diff(np.sign(ecg)))
    crossovers = np.argwhere(signs == -1).flatten()
    # print crossovers
    start = crossovers[0]
    noise = []
    for i in range(1,len(crossovers)):
        per = crossovers[i] - start
        if per >= 50:
            end = crossovers[i]
            break
        else:
            noise.append(crossovers[i])
    ecg = ecg[:2*per]
    # print start, end
    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-3:]  # Finds 9 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ecg.size, d=1.)
    freq_main = np.fft.rfftfreq(ecg.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    # period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ft[:ft_max] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:per])
    #start = 0
    end = start + per
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency
    sample_ += 27.
    sample_ /= 51.
    # sample_double =ecg[start:end + (2 * period)]
    metadata_ = np.array([nu,x,y])
    dump = np.concatenate([metadata_,sample_])
    if min != None:
        if len(dump) < min:
            temp = np.zeros(min - int(len(dump)))
            temp[:] = np.nan
            dump = np.concatenate([dump,temp])

    return dump


def process_multi_feature(T):
    T = T.astype('float')
    centre = T[5]
    mean = np.mean(T,0)
    std = np.mean(T,0)
    kurtosis = stats.kurtosis(T,0)
    skewness = stats.skew(T,0)

    #Vertical Singles
    v63,v74,v85,v30,v41,v52 = T[6]-T[3],T[7]-T[4],T[8]-T[5],T[3]-T[0],T[4]-T[1],T[5]-T[2]
    vs_bar = ( v63 + v74 + v85 + v30 + v41 + v52 )/6
    vs_centreratio = ( v63 + v74 + v85 ) -( v30 + v41 + v52 )
    #Vertical Fulls
    v60,v71,v82 = T[6]-T[0],T[7]-T[1],T[8]-T[2]
    vf_bar = ( v60 + v71 + v82)/3
    #Hortizontal Singles
    h87,h76,h54,h43,h21,h10 = T[8]-T[7],T[7]-T[6],T[5]-T[4],T[4]-T[3],T[2]-T[1],T[1]-T[0]
    hs_bar = ( h87 + h76 + h54 + h43 + h21 + h10 )/6
    hs_centreratio = ( h87 + h76 + h54 ) -( h43 + h21 + h10 )
    #Horizontal Fulls
    h86,h53,h20 = T[8]-T[6],T[5]-T[3],T[2]-T[0]
    hf_bar = ( h86 + h53 + h20 ) / 3
    #Diag11s
    d73,d84,d40,d51 = T[7]-T[3],T[8]-T[4],T[4]-T[0],T[5]-T[1]
    d11_bar = ( d73 + d84 + d40 + d51 ) / 4
    #Diag12s
    d70,d81 = T[7]-T[0],T[8]-T[1]
    d12_bar = ( d70 + d81 ) / 2
    #Diag21s
    d50,d83 = T[5]-T[0],T[8]-T[3]
    d21_bar = ( d50 + d83 ) / 2
    #Diagneg11s
    d64,d42,d31,d75 = T[6]-T[4],T[4]-T[2],T[3]-T[1],T[7]-T[5]
    dn11_bar = ( d64 + d42 + d31 + d75 ) / 4
    #Diagneg12s
    d72,d61 = T[7]-T[2],T[6]-T[1]
    dn12_bar = ( d72 + d61 ) / 2
    #Diagneg21s
    d32,d65 = T[3]-T[2],T[6]-T[5]
    dn21_bar = ( d32 + d65 ) / 2

    axisfocus = (( h54 + v74 ) - ( h43 + v41 )) / 4
    diagfocus = ((d84 + d64) - (d42 + d40)) / 4
    focusratio = axisfocus - diagfocus
    focus = ( axisfocus + diagfocus ) / 2
    total = np.concatenate([centre,mean,std,kurtosis,skewness,vs_bar,vs_centreratio,vf_bar,hs_bar,hs_centreratio,hf_bar,d11_bar,d12_bar,d21_bar,dn11_bar,dn12_bar,dn21_bar,axisfocus,diagfocus,focusratio,focus])
    return total

def sign_solver(Start, half_period = 30.):
    T = Start.astype('float')
    v63,v74,v85,v30,v41,v52 = T[6]-T[3],T[7]-T[4],T[8]-T[5],T[3]-T[0],T[4]-T[1],T[5]-T[2]
    h87,h76,h54,h43,h21,h10 = T[8]-T[7],T[7]-T[6],T[5]-T[4],T[4]-T[3],T[2]-T[1],T[1]-T[0]

    v63_copy1,v74_copy1,v85_copy1,v30_copy1,v41_copy1,v52_copy1 = np.copy(v63),np.copy(v74),np.copy(v85),np.copy(v30),np.copy(v41),np.copy(v52)
    v63_copy2,v74_copy2,v85_copy2,v30_copy2,v41_copy2,v52_copy2 = np.copy(v63),np.copy(v74),np.copy(v85),np.copy(v30),np.copy(v41),np.copy(v52)

    v63_copy1[v63_copy1 > half_period] -= (2*half_period)
    v74_copy1[v74_copy1 > half_period] -= (2*half_period)
    v85_copy1[v85_copy1 > half_period] -= (2*half_period)
    v30_copy1[v30_copy1 > half_period] -= (2*half_period)
    v41_copy1[v41_copy1 > half_period] -= (2*half_period)
    v52_copy1[v52_copy1 > half_period] -= (2*half_period)

    v63_copy2[v63_copy2 < -half_period] %= (2*half_period)
    v74_copy2[v74_copy2 < -half_period] %= (2*half_period)
    v85_copy2[v85_copy2 < -half_period] %= (2*half_period)
    v30_copy2[v30_copy2 < -half_period] %= (2*half_period)
    v41_copy2[v41_copy2 < -half_period] %= (2*half_period)
    v52_copy2[v52_copy2 < -half_period] %= (2*half_period)

    v63_sign = np.sign(v63 * np.sign(v63_copy1) * np.sign(v63_copy2))
    v74_sign = np.sign(v74 * np.sign(v74_copy1) * np.sign(v74_copy2))
    v85_sign = np.sign(v85 * np.sign(v85_copy1) * np.sign(v85_copy2))
    v30_sign = np.sign(v30 * np.sign(v30_copy1) * np.sign(v30_copy2))
    v41_sign = np.sign(v41 * np.sign(v41_copy1) * np.sign(v41_copy2))
    v52_sign = np.sign(v52 * np.sign(v52_copy1) * np.sign(v52_copy2))

    v63_bsign = v63 * np.sign(v63_copy1) * np.sign(v63_copy2)
    v74_bsign = v74 * np.sign(v74_copy1) * np.sign(v74_copy2)
    v85_bsign = v85 * np.sign(v85_copy1) * np.sign(v85_copy2)
    v30_bsign = v30 * np.sign(v30_copy1) * np.sign(v30_copy2)
    v41_bsign = v41 * np.sign(v41_copy1) * np.sign(v41_copy2)
    v52_bsign = v52 * np.sign(v52_copy1) * np.sign(v52_copy2)

    v_sign = (v63_sign +v74_sign +v85_sign +v30_sign +v41_sign +v52_sign )/6
    b_sum = ((v63_bsign +v74_bsign +v85_bsign) / 3) - ((v30_bsign +v41_bsign +v52_bsign ) / 3)
    b_sign = ((v63_sign +v74_sign +v85_sign) / 3) - ((v30_sign +v41_sign +v52_sign ) / 3)


    h87_copy1,h76_copy1,h54_copy1,h43_copy1,h21_copy1,h10_copy1 = np.copy(h87),np.copy(h76),np.copy(h54),np.copy(h43),np.copy(h21),np.copy(h10)
    h87_copy2,h76_copy2,h54_copy2,h43_copy2,h21_copy2,h10_copy2 = np.copy(h87),np.copy(h76),np.copy(h54),np.copy(h43),np.copy(h21),np.copy(h10)

    h87_copy1[h87_copy1 > half_period] -= (2*half_period)
    h76_copy1[h76_copy1 > half_period] -= (2*half_period)
    h54_copy1[h54_copy1 > half_period] -= (2*half_period)
    h43_copy1[h43_copy1 > half_period] -= (2*half_period)
    h21_copy1[h21_copy1 > half_period] -= (2*half_period)
    h10_copy1[h10_copy1 > half_period] -= (2*half_period)

    h87_copy2[h87_copy2 < -half_period] %= (2*half_period)
    h76_copy2[h76_copy2 < -half_period] %= (2*half_period)
    h54_copy2[h54_copy2 < -half_period] %= (2*half_period)
    h43_copy2[h43_copy2 < -half_period] %= (2*half_period)
    h21_copy2[h21_copy2 < -half_period] %= (2*half_period)
    h10_copy2[h10_copy2 < -half_period] %= (2*half_period)

    h87_sign = np.sign(h87 * np.sign(h87_copy1) * np.sign(h87_copy2))
    h76_sign = np.sign(h76 * np.sign(h76_copy1) * np.sign(h76_copy2))
    h54_sign = np.sign(h54 * np.sign(h54_copy1) * np.sign(h54_copy2))
    h43_sign = np.sign(h43 * np.sign(h43_copy1) * np.sign(h43_copy2))
    h21_sign = np.sign(h21 * np.sign(h21_copy1) * np.sign(h21_copy2))
    h10_sign = np.sign(h10 * np.sign(h10_copy1) * np.sign(h10_copy2))

    h_sign = (h87_sign +h76_sign +h54_sign +h43_sign +h21_sign +h10_sign )/6
    axes_sign = np.absolute(v_sign) + np.absolute(h_sign)
    return np.array([v_sign, h_sign, axes_sign, b_sign, b_sum])


def multi_feature_compile(dataframe,test_key = 'Multi Target 0'):
    dataframe = dataframe.copy()
    feature_list = []
    ran = len(dataframe)
    index = np.arange(0,ran,9)

    metadata = dataframe[['Target 0', 'Multi Target 0', 'Vector X 0', 'Vector Y 0','Theta 0', 'Distance 0', 'Nu']][4::9]
    probe_features = ['Crit Position', 'Crit Position 0', 'Crit Position 1', 'Probe Position',
                      'Unit Vector X 0', 'Unit Vector X 1', 'Unit Vector Y 0','Target 0', 'Multi Target 0',
                      'Unit Vector Y 1', 'Theta',  'Theta 1', 'Probe Number','Theta 0',
                      'Nearest Crit Position','Vector X 0', 'Vector Y 0', 'Distance 0', 'Nu']
    all_features = list(dataframe.columns)
    for feature in probe_features:
        if feature in all_features:
            del dataframe['%s' % feature]

    for i in index:
        T = dataframe.iloc[i:i+9].as_matrix()
        S =  dataframe.iloc[i:i+9]['Start'].as_matrix()
        X = process_multi_feature(T)
        Y = sign_solver(S)
        # print i, np.shape(T), np.shape(X)
        feature_list.append(np.concatenate([X,Y]))
    X = np.vstack(feature_list)
    X[X == np.inf] = np.nan
    prefixes = ['Probe Centre: ','Mean Probes: ' ,'Std Probes: ','Kurtosis Probes: ','Skewness Probes' ,'Vertical Singles Mean: ','Vertical Singles Centre Ratio: ','Vertical Full Mean: ','Horizontal Singles Mean: ',
    'Horizontal Singles Centre Ratio: ','Horizontal Full Mean: ','Diagonal 11 Mean: ','Diagonal 12 Mean: ','Diagonal 21 Mean: ','Diagonal N11 Mean: ','Diagonal N12 Mean: ','Diagonal N21 Mean: ','Focus Axis Orientation: ',
    'Focus Diag Orientation: ','Focus Ratio: ','Focus: ']#,'V63: ','V74: ','V85: ','V30: ','V41: ','V52: ','H87: ','H76: ','H54: ','H43: ', 'H21: ','H10: ']
    suffixes = dataframe.keys().tolist()
    keys = []
    for i in prefixes:
        for j in suffixes:
            temp = i + j
            keys.append(temp)
    keys = keys + ['V Sign', 'H Sign', 'Axes Sign']
    print np.shape(metadata)
    df = pd.DataFrame(X,columns = keys)
    df = df.join(metadata.reset_index())
    df.fillna(99999.)

    name = raw_input("Save Filename: ")
    df.to_hdf(name + '.h5','w')


def multi_feature_compile_rt(uncompiled, sign="record_sign_plus"):
    """
    Compiles all the ecg features into a single multi probe feature array. This is then fed into a RF model.
    :param uncompiled: Uncompiled ecg features.
    :param sign: Flag to work out the sign features.
    :return:
    """
    # Compiles all the features
    compiled = process_multi_feature(uncompiled)
    if sign == "record_sign" or sign == "record_sign_plus":
        signs = sign_solver(uncompiled[:, 0])
        compiled = np.concatenate([compiled,signs[:-2]])
    # Cleans all inf values
    # print np.shape(compiled)
    compiled[compiled == np.inf] = 99999.
    if np.isinf(compiled).any():
        print "Infinity came through"
    if np.isnan(compiled).any():
        print "NAN came through"
    if sign == "record_sign_plus":
        return np.nan_to_num(compiled),signs
    else:
        return np.nan_to_num(compiled)


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt3.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names, filled=True, rounded=True)

    command = ["dot", "-Tpdf", "dt3.dot", "-o", "dt3.pdf"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def print_counter(iteration, total, prefix='Counter', suffix='Rotors complete'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    sys.stdout.write('\r%s: %s/%s %s' % (prefix, iteration, total, suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def polar_feature(X, feature, title, rmax=None, clim=None, condition=None):

    plt.figure(figsize=(8.5, 8.5))
    if condition == None:
        r = np.array(X[feature])
        d = np.array(X['Distance 0'])
        theta = np.array(X['Theta 0'])
    else:
        r = np.array(X[feature])[condition]
        d = np.array(X['Distance 0'])[condition]
        theta = np.array(X['Theta 0'])[condition]
    if np.max(r) < 0:
        r = np.absolute(r)

    cm = plt.cm.get_cmap('coolwarm')
    ax = plt.subplot(111, projection='polar')
    PCM = ax.scatter(theta, r, c=d,  marker='.', s=10., edgecolors='none', alpha=0.98, cmap=cm)
    cbar = plt.colorbar(PCM, ax=ax, shrink=0.6, pad=0.07)
    if clim != None:
        PCM.set_clim(vmin=clim[0], vmax=clim[1])
    ax.set_rmin(np.min(r) - 1)
    if rmax != None:
        ax.set_rmax(rmax)
        ax.set_rticks([int(np.min(r) - 1), (int(rmax) + int(np.min(r) - 1))/2, int(rmax)])
    else:
        ax.set_rmax(int(np.max(r) + 1))
        ax.set_rticks([int(np.min(r) - 1), (int(np.min(r) - 1) + (int(np.max(r) + 1)))/2, int(np.max(r) + 1)])
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)
    ax.grid(True)
    ax.set_title(title, va='bottom', fontsize = 20)
    plt.show()

def fcplot(X, feature, clim = None, condition = None):
    plt.figure(figsize =(8.5,8.5))
    if condition == None:
        rad = np.array(X['Distance 0'])
        theta = np.array(X['Theta 0'])
        fea = np.array(X[feature])
    else:
        rad = np.array(X['Distance 0'])[condition]
        theta = np.array(X['Theta 0'])[condition]
        fea = np.array(X[feature])[condition]

    rad = rad[np.logical_not(np.isnan(theta))]
    fea = fea[np.logical_not(np.isnan(theta))]
    theta = theta[np.logical_not(np.isnan(theta))]

    x = rad * np.cos(theta)
    y = rad * np.sin(theta)

    cm = plt.cm.get_cmap('gist_ncar')
    ax = plt.subplot()
    PCM = ax.scatter(x, y, c=fea,  marker = '.', s = 10.,edgecolors = 'none', alpha = 0.98, cmap = cm)
    if clim != None:
        PCM.set_clim(vmin = clim[0],vmax = clim[1])
    cbar = plt.colorbar(PCM, ax = ax, shrink=0.4, pad = 0.07)
    plt.xlim([-200,200])
    plt.ylim([-100,100])
    plt.axes().set_aspect('equal')

    ax.grid(True)
    ax.set_title(str(feature), va='bottom', fontsize = 20)
    plt.show()
    # return x,y,fea

def binplot(X, feature, clim = None, condition = None, binsize = 1, split = 'none', save=False):

    try:
        d = X['Distance 0']
        t = X['Theta 0']
    except:
        d = X['Distance']
        t = X['Theta']
    if condition == None:
        rad = np.array(d)
        theta = np.array(t)
        f = np.array(X[feature])

        # p = np.array(X['Probe Position'])
        # if split == 'mid':
        #     p = (p%200 > 30) * (p%200 < 170)
        # else:
        #     p = (p%200 < 30) + (p%200 > 170)

    else:
        rad = np.array(d)[condition]
        theta = np.array(t)[condition]
        f = np.array(X[feature])[condition]

        # p = np.array(X['Probe Position'])[condition]
        # if split == 'mid':
        #     p = (p%200 > 30) * (p%200 < 170)
        # else:
        #     p = (p%200 < 30) + (p%200 > 170)

    # if split == 'mid' or split == 'out':
    #     rad =  rad[p]
    #     theta = theta[p]
    #     f = f[p]

    rad = rad[np.logical_not(np.isnan(theta))]
    f = f[np.logical_not(np.isnan(theta))]
    theta = theta[np.logical_not(np.isnan(theta))]


    x = rad * np.cos(theta)
    y = rad * np.sin(theta)

    x = x.astype('int')
    y = y.astype('int')
    x /= binsize
    y /= binsize
    x += np.absolute(np.min(x))
    y += np.absolute(np.min(y))
    z = np.zeros((np.max(y) + 1, np.max(x) + 1), dtype = 'float')
    count = np.copy(z)

    for i in range(len(x)):
        z[y[i]][x[i]] += float(f[i])
        count[y[i]][x[i]] += 1.
    z /= count


    # y_grad, x_grad = np.gradient(z)
    cm = plt.cm.get_cmap('coolwarm')
    # plt.figure()
    # plt.imshow(x_grad, interpolation="nearest", origin="lower", cmap = cm)
    # plt.colorbar(shrink=0.4, pad = 0.07)
    # plt.figure()
    # plt.imshow(y_grad, interpolation="nearest", origin="lower", cmap = cm)
    # plt.colorbar(shrink=0.4, pad = 0.07)

    if clim == None:
        clim = [np.nanmin(z),np.nanmax(z)]
        # print(clim)
    plt.figure(figsize =(10.,10.))
    plt.imshow(z,vmin = clim[0],vmax = clim[1], interpolation="nearest", origin="lower", cmap = cm)
    plt.colorbar(shrink=0.4, pad = 0.07)
    plt.xlabel('x', fontsize = 18)
    plt.ylabel('y', fontsize = 18)
    # plt.title(feature, fontsize = 18)
    if save:
        words = [feature]
        words = [w.replace(':', '_') for w in words]
        words = [w.replace(' ', '_') for w in words]
        print words
        plt.savefig(words[0] + '.png')
        plt.close()
    else:
        print(np.shape(z))
        plt.show()

    return z#, clim, feature

def binplot_pretty(X, feature, clim = None, title = 'Default', binsize = 1, split = 'none', save=False):

    try:
        d = X['Distance 0']
        t = X['Theta 0']
    except:
        d = X['Distance']
        t = X['Theta']

    rad = np.array(d)
    theta = np.array(t)
    f = np.array(X[feature])

    rad = rad[np.logical_not(np.isnan(theta))]
    f = f[np.logical_not(np.isnan(theta))]
    theta = theta[np.logical_not(np.isnan(theta))]


    x = rad * np.cos(theta)
    y = rad * np.sin(theta)

    x = x.astype('float')
    y = y.astype('float')
    x /= binsize
    y /= binsize
    x += np.absolute(np.min(x))
    y += np.absolute(np.min(y))
    x = x.astype('int')
    y = y.astype('int')
    z = np.zeros((np.max(y) + 1, np.max(x) + 1), dtype = 'float')
    count = np.copy(z)

    for i in range(len(x)):
        z[y[i]][x[i]] += float(f[i])
        count[y[i]][x[i]] += 1.
    z /= count


    cm = plt.cm.get_cmap('coolwarm')

    if clim == None:
        clim = [np.nanmin(z),np.nanmax(z)]
        # print(clim)
    plt.figure(figsize =(10.,10.))
    plt.imshow(z,vmin = clim[0],vmax = clim[1], interpolation="nearest", origin="lower", cmap = cm,extent=[-180, 180, -99, 100])
    cbar = plt.colorbar(shrink=0.4, pad = 0.07)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel('X', fontsize = 18)
    plt.ylabel('Y', fontsize = 18)
    plt.title(title, fontsize = 21)
    plt.tick_params(axis='both', which='major', labelsize=15)
    if save:
        save_data_name = raw_input("Saved datafile name: ")
        plt.savefig(save_data_name + '.pdf')
        plt.close()
    else:
        print(np.shape(z))
        plt.show()

    return z#, clim, feature


def prob_bins(X, feature, binsize = 1):

    x = np.array(X['Vector X 0'])
    y = np.array(X['Vector Y 0'])
    f = np.array(X[feature])
    f += 1.
    f *= 6.
    f = np.rint(f)

    x = x.astype('float')
    y = y.astype('float')
    x /= binsize
    y /= binsize
    x += np.absolute(np.min(x))
    y += np.absolute(np.min(y))
    x = x.astype('int')
    y = y.astype('int')
    z = np.zeros((np.max(y) + 1, np.max(x) + 1), dtype = 'float')
    p = np.zeros((np.max(y) + 1, np.max(x) + 1,13), dtype = 'float')
    c = np.zeros(13)
    count = np.copy(z)

    for i in range(len(x)):
        p[y[i]][x[i]][f[i]] += 1.
        count[y[i]][x[i]] += 1.
        c[f[i]] += 1.
    count = count.reshape((len(count[:,0]),len(count[0]),1))
    p /= count
    print c
    return p#, clim, feature

def check_signs(x,y,sign_value,sign_tensor,thr = 0.1):
    x = float(x) / 10.
    y = float(y) / 10.
    x = int(x + 18.)
    y = int(y + 10.)
    j = int((sign_value + 1.) * 6.)
    p = sign_tensor[:,:,j][y][x]
    # print x,y,j,p
    return p > thr

def check_bsign(bsign,bsum,thr = -10):
    if bsign < 0 or bsum < thr:
        return bsum
    else:
        return 0

def plot_matrix(z,title):
    cm = plt.cm.get_cmap('coolwarm')
    plt.figure(figsize =(10.,10.))
    plt.imshow(z, interpolation="nearest", origin="lower", cmap = cm,extent=[-180, 180, -99, 100])
    cbar = plt.colorbar(shrink=0.4, pad = 0.07)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel('X', fontsize = 18)
    plt.ylabel('Y', fontsize = 18)
    plt.title(title, fontsize = 21)
    plt.tick_params(axis='both', which='major', labelsize=15)
    save_data_name = raw_input("Saved datafile name: ")
    plt.savefig(save_data_name + '.pdf')
    plt.close()

def modeplot(X, feature, clim = None, condition = None, binsize = 1, split = 'none', save=False):

    try:
        d = X['Distance 0']
        t = X['Theta 0']
    except:
        d = X['Distance']
        t = X['Theta']
    if condition == None:
        rad = np.array(d)
        theta = np.array(t)
        f = np.array(X[feature])

    else:
        rad = np.array(d)[condition]
        theta = np.array(t)[condition]
        f = np.array(X[feature])[condition]

    rad = rad[np.logical_not(np.isnan(theta))]
    f = f[np.logical_not(np.isnan(theta))]
    theta = theta[np.logical_not(np.isnan(theta))]

    x = rad * np.cos(theta)
    y = rad * np.sin(theta)

    x = x.astype('int')
    y = y.astype('int')
    x /= binsize
    y /= binsize
    x += np.absolute(np.min(x))
    y += np.absolute(np.min(y))
    z = [ [[] for _ in range(np.max(x)+1)] for _ in range(np.max(y)+1)]
    count = np.zeros((np.max(y) + 1, np.max(x) + 1))

    for i in range(len(x)):
        z[y[i]][x[i]] += [float(f[i])]
        count[y[i]][x[i]] = 1.

    for index_1, value_1 in enumerate(z):
        for index_2, value_2 in enumerate(value_1):
            if value_2:
                z[index_1][index_2] = mode(value_2)[0][0]
            else:
                z[index_1][index_2] = 0.0

    z /= count

    if clim == None:
        clim = [np.nanmin(z),np.nanmax(z)]

    return z, clim, feature


def feature_prune(dataframe, delete_list):
    """
    Function that will delete the desired features from pandas dataframe before ML model/analysis is made.
    :param dataframe: Pandas dataframe (Required)
    :param delete_list: List (Required)
    :return:
    """
    for column in delete_list:
        del dataframe['%s' % column]


def target_distance_creation(row):
    """
    For use with the 2 beat case (Creates a column of the True distance from the ECG probe.)
    :param row: row of the dataframe (Required)
    :return:
    """
    nearest_cp = str(int(row['Nearest Crit Position']))
    return row['Distance %s' % nearest_cp]


def target_creation(row):
    """
    For use with the 2 beat case (Creates a column for the target.)
    :param row: row of the dataframe (Required)
    :return:
    """
    nearest_cp = str(int(row['Nearest Crit Position']))
    if row['Distance %s' % nearest_cp] <= np.sqrt(200):
        return 1
    else:
        return 0


def distance(x_vectors, y_vectors, y_scale=1, x_scale=1):
    # Old method for re-scaling
    """
    Function used to rework out the distance in a pandas dataframe (use in ipython). Need to convert the columns into
    tuples via np.unravel_index.
    :param col1: df['Crit Position 0']
    :param col2: df['Probe Position']
    :param y_scale: scaling for y
    :param x_scale: sacling for x
    :return: New distance
    """

    # col1_values = [int(x) for x in col1.values]
    # col2_values = [int(x) for x in col2.values]
    # col1_t = [np.unravel_index(index, dims=(200, 200)) for index in col1_values]
    # col2_t = [np.unravel_index(index, dims=(200, 200)) for index in col2_values]
    # x_vectors = [(value1[1]-value2[1]) for value1, value2 in zip(col1_t,col2_t)]
    # y_vectors = [(value1[0]-value2[0]) for value1, value2 in zip(col1_t,col2_t)]
    # y_vectors = [y-200 if y>100 else y+200 if y<=-100 else y for y in y_vectors]
    # return [np.sqrt((y_scale*y)**2 + (x_scale*x)**2) for y, x in zip(y_vectors,x_vectors)]

    """
<<<<<<< HEAD
    :param x_vectors: df['Vector X 0']
    :param y_vectors: df['Vector Y 0']
    :param y_scale: scaling for y
    :param x_scale: sacling for x
    :return: New distance
    """

    return [np.sqrt((y_scale*y)**2 + (x_scale*x)**2) for y, x in zip(y_vectors.values,x_vectors.values)]


def y_vector_classifier(x):
    """
    :param x: pandas series
    :param threshold:
    :return: turns vector data into classifier information.
    """
    if np.abs(x) >= 3:
        return 0
    if np.abs(x) < 3:
        return 1

def x_vector_classifier(x):
    """
    :param x: pandas series
    :param threshold:
    :return: turns vector data into classifier information.
    """
    if np.abs(x) >= 8:
        return 0
    if np.abs(x) < 8:
        return 1

def winsum(interval, window_size):
    window = np.ones(int(window_size))
    return np.convolve(interval, window, 'same')

def ypredictor(probs,thr = 0.3):
    ws = 2
    xs = np.arange(-99,101)
    while np.max(probs) < thr:
        probs = winsum(probs,ws)
        ws += 1
        if ws == 5:
            nz = xs[probs != 0]
            m = np.mean(nz)
            if m > 0:
                m = np.mean(nz[nz>0])
                return int(m)
            else:
                m = np.mean(nz[nz<=0])
                return int(m)
    return xs[np.argmax(probs)]
    # p = xs[probs == np.max(probs)]
    # if len(p) == 1:
    #     return np.argmax(probs)
    # else:
    #     if np.all(p>0) or np.all(p<0):
    #         pred = int(np.mean(p))
    #         return pred
    #     else:
    #         return np.argmax(probs)
