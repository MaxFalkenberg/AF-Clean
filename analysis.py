import numpy as np
import propagate
import cPickle


def af_starts(start, end):
    nu = []
    af_initiated = []
    af_start_time = []
    for i in np.arange(start, end, 0.005):
        temp1 = []
        temp2 = []
        nu.append(i)
        print 'nu = ', i
        for j in range(1):
            a = propagate.Heart(nu=i, delta=0.05, eps=0.05, rp=50, count_excited='start', print_t=False)
            a.set_pulse(220)
            x, y = a.propagate(1000000)
            temp1.append(x)
            temp2.append(y)
        af_initiated.append(temp1)
        af_start_time.append(temp2)
    return nu, af_initiated, af_start_time


def af_duration(nu_list):
    for i in nu_list:
        print 'nu = ', i
        in_af_temp = []
        mean_time_temp = []
        exc_cell_temp = []
        for j in range(1):
            a = propagate.Heart(nu=i, delta=0.05, eps=0.05, rp=50, count_excited='time', print_t=False)
            a.set_pulse(220)
            x, y = a.propagate(1000000)
            in_af_temp.append(np.array(x))
            mean_time_temp.append(np.mean(x))
            exc_cell_temp.append(np.array(y))
        z = (['nu =' + str(i) + ' , delta = 0.05,eps = 0.05,rp = 50'], in_af_temp, mean_time_temp, exc_cell_temp)
        try:
            save('af_duration_data_nu' + str(i)[2:], z)
        except:
            return z


def save(filename, obj):
    """Use to save pile object to chosen directory."""
    cPickle.dump(obj, open(str(filename)+".pickle", 'wb'))


def load(filename):
    """Load pile object (or other pickle file) from chosen directory."""
    return cPickle.load(open(str(filename)+".pickle", 'rb'))
    return (nu, in_af, mean_time_in_af, exc_cell_count)
