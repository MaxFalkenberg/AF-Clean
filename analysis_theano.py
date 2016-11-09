import numpy as np
import propagate
import cPickle
import theano.tensor as T
from theano import function
from theano.tensor.signal.conv import conv2d
from itertools import product


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

def course_grain(excitation_grid, cg_factor):
    """ excitation_grid should be list of 2d arrays in time order where each 2d array
    is the animation state of the system at time t. The excitation_grid
    of a system can be obtained  using b = animator.Visual('file_name'), selecting your
    desired animation range and then exporting excitation_grid = b.animation_data.

    cg_factor is the unitless factor corresponding to the number of small original cells
    along each side of the new course grained cell.
    e.g. If a 200x200 array is processed with cg_factor = 5, the new course grained array
    will be shape 40x40 where each new cell corresponds to the net excitations from 5x5
    sections of the original array."""

    exc = np.array(excitation_grid).astype('float') #Asserts data type of imported excitation_grid
    filt = np.ones((cg_factor,cg_factor),dtype = 'float') #Square matrix of ones in shape of course_grained cells.
    norm = cg_factor ** 2 #Number of original cells in each course grained cell
    a = T.dtensor3('a') #Theano requires us to specify data types. dtensor3 is a 3d tensor of float64's
    b = T.dmatrix('b') #Matrix of float64's
    z = conv2d(a,b,subsample = (cg_factor,cg_factor)) / norm #This specifies the function to process.
    #               Convolution with subsample step length results in course grained matrices
    f = function([a,b],z) #Theano function definition where inputs ([a,b]) and outputs (z) are specified
    return f(exc,filt) #Returns function with excitation_grid and filter as output

class ECG:

    def __init__(self, shape = (200,200), probe_height = 3):
        self.shape = shape
        self.y_mid = self.shape[0]/2
        self.y_mid = np.array(self.y_mid, dtype = 'int32')
        self.z = probe_height
        self.probe_position = None

        print '[r,s,c (have to set in code)]'
        mode = str(raw_input('Ecg position mode: '))

        self.mode = mode

        if mode == 'r':
            electrode_spacing = int(raw_input('Choose Electrode Spacing: '))
            self.electrode_spacing = electrode_spacing
            self.probe_y = np.arange(electrode_spacing - 1,self.shape[0],electrode_spacing, dtype = 'float32')
            self.probe_x = np.arange(electrode_spacing - 1,self.shape[1],electrode_spacing, dtype = 'float32') - (electrode_spacing/2)
            self.probe_position = list(product(self.probe_y, self.probe_x))
        if mode == 's':
            y = raw_input('Electrode y position:')
            x = raw_input('Electrode x position:')
            self.probe_y = np.array([y],dtype = 'int32')
            self.probe_x = np.array([x],dtype = 'int32')
        if mode == 'c':
            self.probe_y = np.array([99], dtype='int32')
            self.probe_x = np.arange(10, 200, 10, dtype='int32')

        self.base_y_x = np.zeros((self.shape[0] - 1,self.shape[1]), dtype = 'float32')
        self.base_y_y = np.zeros((self.shape[0] - 1,self.shape[1]), dtype = 'float32')
        self.base_x_y = np.zeros((self.shape[0],self.shape[1] - 1), dtype = 'float32')
        self.base_x_x = np.zeros((self.shape[0],self.shape[1] - 1), dtype = 'float32')

        for i in range(len(self.base_x_y)):
            self.base_x_y[i] = i
            self.base_y_x[:,i] = i

        for i in range(len(self.base_y_y)):
            self.base_y_y[i] = i
            self.base_x_x[:,i] = i

        self.base_x_y -= self.y_mid
        self.base_y_y[:self.y_mid] -= self.y_mid
        self.base_y_y[self.y_mid:] -= self.y_mid - 1.

        self.shifted_x_x = []
        self.shifted_y_x = []

        for i in self.probe_x:
            self.shifted_y_x.append(self.base_y_x - i)
            temp = np.copy(self.base_x_x)
            temp[:,:int(i)] -= i
            temp[:,int(i):] -= i - 1.
            self.shifted_x_x.append(temp)

        self.ygrad_den = []
        self.xgrad_den = []

        for i in range(len(self.shifted_x_x)):
            self.ygrad_den.append(((self.shifted_y_x[i] ** 2) + (self.base_y_y ** 2) + (self.z ** 2)) ** 1.5)
            self.xgrad_den.append(((self.shifted_x_x[i] ** 2) + (self.base_x_y ** 2) + (self.z ** 2)) ** 1.5)


    def solve(self,inp):
        """inp is 3d numpy array where first dimension is time and remaining dimensions are excitation state of system.
        Current method of import is inp = np.array(b.animation_data) where b = animator.Visual('file_name').

        Return list of arrays. Each array is ECG data for a particular probe."""

        dim = len(np.shape(inp))
        probe_y = T.iscalar('probe_y')
        mid = T.iscalar('mid')
        if dim == 2:
            inp = inp.reshape((1,np.shape(inp)[0],np.shape(inp)[0]))
            dim = 3
        if dim != 3:
            raise ValueError("Input data wrong dimension.")
        else:
            grid = T.ftensor3('grid')
            roll = T.roll(grid,mid - probe_y,axis = 1)

            rolled_grid = T.ftensor3('rolled_grid')
            y_grad = T.extra_ops.diff(rolled_grid,axis = 1)
            x_grad = T.extra_ops.diff(rolled_grid,axis = 2)

        F_roll = function([grid,mid,probe_y], roll)
        F_ygrad = function([rolled_grid], y_grad)
        F_xgrad = function([rolled_grid], x_grad)

        dif = T.fmatrix('dif')
        den = T.fmatrix('den')
        grad = T.ftensor3('grad')
        subtot = (dif * grad) / den
        F_subtot = function([grad,dif,den],subtot)

        x_subtot = T.ftensor3('x_subtot')
        y_subtot = T.ftensor3('y_subtot')
        tot = T.sum(x_subtot, axis = [1,2]) + T.sum(y_subtot, axis = [1,2])
        F_tot = function([x_subtot,y_subtot],tot)
        ECG_values = []
        # theano.printing.pydotprint(F_tot, outfile = 'theano.png')


        for i in range(len(self.probe_y)):
            Rolled_Grid = F_roll(inp,self.y_mid,self.probe_y[i])
            self.r = Rolled_Grid
            Y_grad = F_ygrad(Rolled_Grid)
            X_grad = F_xgrad(Rolled_Grid)
            for j in range(len(self.probe_x)):
                ECG_values.append(F_tot(F_subtot(X_grad,self.shifted_x_x[j],self.xgrad_den[j]),F_subtot(Y_grad,self.base_y_y,self.ygrad_den[i])))

        return ECG_values

def ecg_data(excitation_grid, cg_factor, probe_pos = None): #By default probe at (shape[0]/2,shape[1]/2)
    """Returns ECG time series from excitation grid which is list of system state matrix at
    each time step. This can either come from b = animator.Visual('file_name') -> b.animation_data,
    or can be course grained using 'course_grain' If data has been course grained, this must be
    specified in cg_factor to ensure distance between cells are correctly adjusted. Probe position
    can be specified as a tuple of course grained coordinates ints (y,x). If probe_pos == None, probe
    will be placed in centre of tissue. """

    shape = np.shape(excitation_grid)
    if type(excitation_grid) == list:
        excitation_grid = np.array(excitation_grid)
    exc  = excitation_grid.astype('float')
    # ex = T.dtensor3('ex') #Theano variable definition
    # z1 = 50 - ex #Converts excitation state to time state counter.
    # #i.e. excited state = 0, refractory state 40 -> 50 - 40 = 10
    # z2 = (((((50 - z1) ** 0.3) * T.exp(-(z1**4)/1500000) + T.exp(-z1)) / 4.2) * 110) - 20 #State voltage conversion with theano
    # f = function([ex], z2)
    # exc = f(exc) * (cg_factor ** 2)

    if probe_pos != None:
        #If y coordinate of probe is not in tissue centre,
        #this will roll matrix rows until probe y coordinate is in central row
        exc = np.roll(exc,(shape[1]/2) - probe_pos[0],axis = 1)

    x_dif = np.gradient(exc,axis = 2)
    y_dif = np.gradient(exc,axis = 1)
    x_dist = np.zeros_like(x_dif[0])
    y_dist = np.zeros_like(y_dif[0])

    for i in range(len(x_dist[0])):
        x_dist[:,i] = i
    for i in range(len(y_dist)):
        y_dist[i] = i
    if probe_pos == None:
        x_dist -= (shape[2] / 2)
    else:
        x_dist -= probe_pos[1]
    y_dist -= (shape[1] / 2)

    net_x = x_dist * x_dif
    net_y = y_dist * y_dif
    net = net_x + net_y
    z = 3
    den = (((cg_factor * x_dist) ** 2) + ((cg_factor * y_dist) ** 2) + (z ** 2)) ** 1.5
    ecg_values = []
    for i in range(len(net)):
        try:
            ecg_values.append(np.sum(net[i]/den))
        except:
            pass
    return ecg_values

class ECG_single:

    def __init__(self,shape, probe_height):
        """Class for dynamically returning ECG voltage of a particular excitation state
        at a particular probe position. Initialise before running any animations. """
        self.shape = shape
        self.roll = shape[0] / 2
        self.probe_height = probe_height
        x_dist = np.zeros(shape)
        y_dist = np.zeros(shape)
        for i in range(shape[1]):
            x_dist[:,i] = i
        for i in range(shape[0]):
            y_dist[i] = i
        self.x_dist = x_dist
        self.y_dist = y_dist - self.roll

        self.ex = T.dmatrix('ex') #Theano variable definition
        self.z1 = 50 - self.ex #Converts excitation state to time state counter.
        self.z2 = (((((50 - self.z1) ** 0.3) * T.exp(-(self.z1**4)/1500000) + T.exp(-self.z1)) / 4.2) * 110) - 20
        self.f = function([self.ex], self.z2)

        self.xd = T.dmatrix('xd')
        self.yd = T.dmatrix('yd')
        self.den = (((self.xd) ** 2) + ((self.yd) ** 2) + (self.probe_height ** 2)) ** 1.5
        self.g = function([self.xd,self.yd],self.den)

    def voltage(self,excitation_matrix, probe_centre):
        """excitation_matrix is current system excited state matrix imported from animator.Visual.animation_data.
        Probe centre should be entered as a tuple (y,x)"""
        if type(excitation_matrix) == list:
            excitation_matrix = np.array(excitation_matrix)
        voltages = np.roll(self.f(excitation_matrix.astype('float')),self.roll - probe_centre[0],axis = 0)
        x_dif = np.gradient(voltages,axis = 1)
        y_dif = np.gradient(voltages,axis = 0)
        x_temp = self.x_dist - probe_centre[1]
        return np.sum(((x_dif * x_temp) + (y_dif * self.y_dist)) / self.g(x_temp,self.y_dist))


def save(filename, obj):
    """Use to save pile object to chosen directory."""
    cPickle.dump(obj, open(str(filename)+".pickle", 'wb'))


def load(filename):
    """Load pile object (or other pickle file) from chosen directory."""
    return cPickle.load(open(str(filename)+".pickle", 'rb'))
    return (nu, in_af, mean_time_in_af, exc_cell_count)
