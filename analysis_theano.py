import numpy as np
import propagate
import cPickle
import theano.tensor as T
from theano import function
from theano.tensor.signal.conv import conv2d


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


def af_duration(nu_list):
    for i in nu_list:
        print 'nu = ', i
        in_af_temp = []
        mean_time_temp = []
        exc_cell_temp = []
        for j in range(1):
            a = propagate.Heart(nu=i, delta=0.05, eps=0.05, rp=50, count_excited='time', print_t=False)
            a.set_pulse(220)
            x, y = a.propagate(100000)
            in_af_temp.append(np.array(x))
            mean_time_temp.append(np.mean(x))
            exc_cell_temp.append(np.array(y))
        z = (['nu =' + str(i) + ' , delta = 0.05,eps = 0.05,rp = 50'], in_af_temp, mean_time_temp, exc_cell_temp)
        try:
            save('af_duration_data_nu' + str(i)[2:], z)
        except:
            return z

def course_grain(excitation_grid, cg_factor):#file_name, cg_factor):
    exc = np.array(excitation_grid).astype('float')
    filt = np.ones((cg_factor,cg_factor),dtype = 'float')
    norm = cg_factor ** 2
    a = T.dtensor3('a')
    b = T.dmatrix('b')
    z = conv2d(a,b,subsample = (cg_factor,cg_factor)) / norm
    f = function([a,b],z)
    return f(exc,filt)

def ap_curve(grid):
    return (((((50 - grid) ** 0.3) * np.exp(-(grid**4)/1500000) + np.exp(-grid)) / 4.2) * 110) - 20

def ecg_data(excitation_grid, cg_factor, probe_pos = None): #By default probe at (shape[0]/2,shape[1]/2)
    shape = np.shape(excitation_grid)
    exc  = excitation_grid.astype('float')
    ex = T.dtensor3('ex')
    z1 = 50 - ex
    z2 = (((((50 - z1) ** 0.3) * T.exp(-(z1**4)/1500000) + T.exp(-z1)) / 4.2) * 110) - 20
    f = function([ex], z2)
    exc = f(exc)

    if probe_pos != None:
        for i in range(len(exc)):
            exc[i] = np.roll(exc[i],(shape[1]/2) - probe_pos[0],axis = 0)

    x_dif = []
    y_dif = []
    x_zero = np.zeros((shape[2],1), dtype = 'float')
    y_zero = np.zeros((1,shape[1]), dtype = 'float')
    for i in exc:
        x_dif.append(np.append(np.diff(i),x_zero,axis = 1))
        y_dif.append(np.append(np.diff(i,axis = 0),y_zero, axis = 0))
    x_dist = np.zeros((shape[1],shape[2]))
    y_dist = np.zeros((shape[1],shape[2]))

    for i in range(len(x_dist)):
        x_dist[:,i] = i
        y_dist[i] = i
    if probe_pos == None:
        x_dist -= shape[2] / 2
    else:
        x_dist -= probe_pos[1]
    y_dist -= shape[1] / 2

    x_dist += 0.5
    y_dist += 0.5 
    net = []
    z = 3
    den = (((cg_factor * x_dist) ** 2) + ((cg_factor * y_dist) ** 2) + (z ** 2)) ** 1.5
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


def save(filename, obj):
    """Use to save pile object to chosen directory."""
    cPickle.dump(obj, open(str(filename)+".pickle", 'wb'))


def load(filename):
    """Load pile object (or other pickle file) from chosen directory."""
    return cPickle.load(open(str(filename)+".pickle", 'rb'))
    return (nu, in_af, mean_time_in_af, exc_cell_count)
