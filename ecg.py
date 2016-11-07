import numpy as np
import theano
import theano.tensor as T
from theano import function
THEANO_FLAGS = 'config.profile = True, config.profile_memory = True,floatX=float32'

class ECG:

    def __init__(self, shape = (200,200), probe_height = 3):
        self.shape = shape
        self.y_mid = self.shape[0]/2
        self.y_mid = np.array(self.y_mid, dtype = 'int32')
        self.z = probe_height

        mode = str(input('Would you like a range (input = range) of electrodes or a single electrode (input = single)?'))

        if mode == 'range':
            electrode_spacing = int(input('Choose Electrode Spacing: '))
            self.electrode_spacing = electrode_spacing
            self.probe_y = np.arange(electrode_spacing - 1,self.shape[0],electrode_spacing, dtype = 'float32')
            self.probe_x = np.arange(electrode_spacing - 1,self.shape[1],electrode_spacing, dtype = 'float32')
        if mode == 'single':
            y = input('Electrode y position:')
            x = input('Electrode x position:')
            self.probe_y = np.array([y],dtype = 'int32')
            self.probe_x = np.array([x],dtype = 'int32')

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
        inp = inp.astype('float32')
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

        F_roll = function([grid,mid,probe_y], roll,profile = True)
        F_ygrad = function([rolled_grid], y_grad,profile = True)
        F_xgrad = function([rolled_grid], x_grad,profile = True)

        dif = T.fmatrix('dif')
        den = T.fmatrix('den')
        grad = T.ftensor3('grad')
        subtot = (dif * grad) / den
        F_subtot = function([grad,dif,den],subtot,profile = True)

        x_subtot = T.ftensor3('x_subtot')
        y_subtot = T.ftensor3('y_subtot')
        tot = T.sum(x_subtot, axis = [1,2]) + T.sum(y_subtot, axis = [1,2])
        F_tot = function([x_subtot,y_subtot],tot, profile = True)
        # theano.printing.pydotprint(F_subtot, outfile="f.png", var_with_name_simple=True)
        ECG_values = []


        for i in range(len(self.probe_y)):
            Rolled_Grid = F_roll(inp,self.y_mid,self.probe_y[i])
            Y_grad = F_ygrad(Rolled_Grid)
            X_grad = F_xgrad(Rolled_Grid)
            for j in range(len(self.probe_x)):
                ECG_values.append(F_tot(F_subtot(X_grad,self.shifted_x_x[j],self.xgrad_den[j]),F_subtot(Y_grad,self.base_y_y,self.ygrad_den[i])))

        return ECG_values
