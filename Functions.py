import numpy as np
import h5py
import matplotlib.pyplot as plt
import theano.tensor as T
from theano import function
from theano.tensor.signal.conv import conv2d
from itertools import product
from numpy.fft import rfft
from numpy.fft import irfft
from sklearn.tree import export_graphviz
import subprocess
import sys


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
    crit_point = cp #Index of critical point
    probe_point = np.ravel_multi_index(probes.astype('int')[number], (200, 200))
    y,x = np.unravel_index(cp,(200,200))
    # dist = roll_dist(cp)[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    def cp_vector(y_probe,x_probe):
        x_vector = int(x_probe) - x
        y_vector = int(y_probe) - y
        if y_vector > 100:
            y_vector -= 200
        elif y_vector <= -100:
            y_vector += 200

        r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
        c = (x_vector + (1j * y_vector)) /r
        theta = np.angle(c)
        return r,float(x_vector)/r,float(y_vector)/r,theta

    dist, unit_vector_x,unit_vector_y, theta = cp_vector(probes[number][0],probes[number][1])

    if dist <= np.sqrt(200):
        target = 1
    else:
        target = 0

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
    start = np.argmax(ift[:(2*period) - 1])
    end = start + (2 * period)
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.

    ft_samp = rfft(sample_)  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(ft.size, d=1.)
    ft_samp_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_samp_max10 = np.argsort(ft_abs)[-9:]  # Finds 9 largest frequency fundamentals

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
                        [largest_sum, crit_point, probe_point, dist, unit_vector_x, unit_vector_y, theta, target])

    return features


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt2.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names, filled=True, rounded=True)

    command = ["dot", "-Tpdf", "dt2.dot", "-o", "dt2.pdf"]
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


def polar_feature(X, feature, title, rmax=None, clim=None, condition=None):

    plt.figure(figsize=(8.5, 8.5))
    if condition == None:
        r = np.array(X[feature])
        d = np.array(X['Distance'])
        theta = np.array(X['Theta'])
    else:
        r = np.array(X[feature])[condition]
        d = np.array(X['Distance'])[condition]
        theta = np.array(X['Theta'])[condition]
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


def feature_prune(dataframe, delete_list):
    """
    Function that will delete the desired features from pandas dataframe before ML model/analysis is made.
    :param: dataframe (Required): Pandas dataframe
    :param: delete_list (Required): List
    :return:
    """
    for column in delete_list:
        del dataframe['%s' % column]
