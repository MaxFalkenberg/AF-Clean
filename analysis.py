import numpy as np
import propagate
import cPickle
import theano.tensor as T
from theano import function


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
            x, y = a.propagate(100000)
            temp1.append(x)
            temp2.append(y)
        af_initiated.append(temp1)
        af_start_time.append(temp2)
    return nu, af_initiated, af_start_time


def af_duration(nu_list, iterations, length):
    for i in nu_list:
        print 'nu = ', i
        in_af_temp = []
        mean_time_temp = []
        exc_cell_temp = []
        for j in range(iterations):
            a = propagate.Heart(nu=i, delta=0.05, eps=0.05, rp=50, count_excited='time', print_t=False)
            a.set_pulse(220)
            x, y = a.propagate(length)
            in_af_temp.append(np.array(x))
            mean_time_temp.append(np.mean(x))
            exc_cell_temp.append(np.array(y))

        z = (['nu =' + str(i) + ' , delta = 0.05,eps = 0.05,rp = 50'], in_af_temp, mean_time_temp, exc_cell_temp)
        try:
            save('af_duration_data_nu' + str(i)[2:], z)
        except:
            return z

def ap_curve(grid):
    return (((((50 - grid) ** 0.3) * np.exp(-(grid**4)/1500000) + np.exp(-grid)) / 4.2) * 110) - 20

def ecg_data(excitation_grid):
    exc  = excitation_grid.astype('int8')
    # twenty = np.ones((200,200), dtype = 'float')* 20
    # rp = np.ones((200,200), dtype = 'float') * 50

    ex = T.btensor3('ex')
    z1 = 50 - ex
    z2 = (((((50 - z1) ** 0.3) * T.exp(-(z1**4)/1500000) + T.exp(-z1)) / 4.2) * 110) - 20

    f = function([ex], z2)

    print f(exc)
    return

    # z = (((((a - c) ** 0.3) * np.exp(-(c**4)/1500000) + np.exp(-c)) / 4.2) * 110) - b
    # f = function([a,b,c],z)

    for i in range(len(exc)):
        exc[i] = rp - exc[i]
        exc[i] = f(rp,exc[i],twenty)    #ap_curve(exc[i])
    # for i in excitation_grid:
    #     voltage_grid.append(twenty - (110./50) * (rp - i))
    x_dif = []
    y_dif = []
    x_zero = np.zeros((200,1), dtype = 'float')
    y_zero = np.zeros((1,200), dtype = 'float')
    for i in exc:
        x_dif.append(np.append(np.diff(i),x_zero,axis = 1))
        y_dif.append(np.append(np.diff(i,axis = 0),y_zero, axis = 0))
    print np.shape(x_dif[0]),np.shape(y_dif[0])
    x_dist = np.zeros((200,200))
    y_dist = np.zeros((200,200))

    for i in range(len(x_dist)):
        x_dist[:,i] = i
        y_dist[i] = i
    x_dist -= 100
    y_dist -= 100
    net = []
    z = 3
    den = ((x_dist ** 2) + (y_dist ** 2) + (z ** 2)) ** 1.5
    #den += 1
    for i in range(len(x_dif)):
        net.append(x_dist * x_dif[i] + y_dist * y_dif[i])
    ecg_values = []
    for i in range(len(net)):
        try:
            ecg_values.append(np.sum(net[i]/den))
        except:
            pass

    return ecg_values


def delta_sweep(delta_list, iterations, length):
    for i in delta_list:
        print 'delta = %s' % i
        in_af_temp = []
        mean_time_temp = []
        exc_cell_temp = []
        for j in range(iterations):
            a = propagate.Heart(nu=0.13, delta=i, eps=0.05, rp=50, seed=None, count_excited='time', print_t=False)
            a.set_pulse(220)
            x, y = a.propagate(length)
            in_af_temp.append(np.array(x))
            mean_time_temp.append(np.mean(x))
            exc_cell_temp.append(np.array(y))
            print mean_time_temp
            print exc_cell_temp


def save(filename, obj):
    """Use to save pile object to chosen directory."""
    cPickle.dump(obj, open(str(filename)+".pickle", 'wb'))


def load(filename):
    """Load pile object (or other pickle file) from chosen directory."""
    return cPickle.load(open(str(filename)+".pickle", 'rb'))
    return (nu, in_af, mean_time_in_af, exc_cell_count)
