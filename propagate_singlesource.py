import numpy as np
# from numba import jit


class Heart:

    def __init__(self, nu=1.0, delta=0., eps=0.2, rp=50, fakedata=False):
        """Fraction of vertical connections given: \'nu\'.
            Vertical connections are randomly filled.
            Fraction of dysfunctional cells: \'delta\'.
            Probability of failed firing: \'eps\'."""

        self.pulse_vectors = None
        self.pulse_index = None
        self.pulse_rate = 220
        self.pulse_norm = None
        self.t = 0
        self.__n = nu  # Private vertical fractions variable
        self.__d = delta  # Private cell dysfunction variable
        self.__e = eps  # Private cell depolarisation failure variable
        self.shape = (200, 200)
        self.size = self.shape[0] * self.shape[1]
        self.rp = rp
        self.excited = []
        self.exc_total = []
        self.starting_t = 0
        self.r_true = (np.arange(self.size) % self.shape[1] != self.shape[1] - 1)
        self.l_true = (np.arange(self.size) % self.shape[1] != 0)
        self.u = np.ones(self.size, dtype = 'int32') * self.shape[1]
        self.u[-self.shape[1]::] = - self.size + self.shape[0]
        self.d = np.ones(self.size, dtype = 'int32') * - self.shape[1]
        self.d[:self.shape[1]:] = + self.size - self.shape[0]
        self.fakedata = fakedata

        self.cell_grid = np.ones(self.size,
                                  dtype='bool')  # Grid on which signal will propagate. Defines whether cell is at rest, excited or refractory.
        self.cell_vert = np.zeros(self.size,
                                  dtype='bool')  # Defines whether cell has vertical connection. 1 = Yes, 0 = No.
        self.cell_dys = np.zeros(self.size, dtype='bool')  # Defines whether cell is dysfunctional. 1 = Yes, 0 = No.
        self.cell_norm = np.invert(self.cell_dys)

        for i in range(self.size):
            rand_nu = np.random.random(1)[0]
            rand_delta = np.random.random(1)[0]

            if rand_nu < self.__n:  # If rand_nu < self.__n, cell (x,y) has connection to (x,y+1)
                if rand_delta < self.__d:  # If rand_delta < self.__d, cell (x,y) is dyfunctional. Failes to fire with P = self.__e.
                    self.cell_vert[i] = True  # Both vertically connected and dysfunctional.
                    self.cell_dys[i] = True
                else:
                    self.cell_vert[i] = True  # Vertically connected but not dysfunctional.
            else:
                if rand_delta < self.__d:  # Dysfunctional but not vertically connected.
                    self.cell_dys[i] = True

        self.cell_norm = np.invert(self.cell_dys)

    def set_pulse(self, rate, vectors=None):
        # Use before self.pulse. Defines the rate at which the pulse fires and if desired
        # defines custom pacemaker cells. If vectors = None, left most wall of tissue will be pacemaker.
        if vectors is not None:
            self.pulse_vectors = vectors
        self.pulse_rate = rate
        self.pulse_norm = True

    def set_doublepulse(self, rate1, vectors1, rate2, vectors2):
        self.pulse_vectors1 = vectors1
        self.pulse_vectors2 = vectors2
        self.pulse_rate1 = rate1
        self.pulse_rate2 = rate2
        self.pulse_norm = False

    def pulse(self):  # Still need to include functionality to avoid exciteing blocked cells
        """If cells = None, column x = 0 will be by defaults excited.
            To set custom cells to excite inputs cells as list of two lists
            with first list as y coordinates and second list as x coordinates.

            i.e. vectors = [[y1,y2,y3...],[x1,x2,x3...]]

            This will excite cells (x1,y1),(x2,y2),(x3,y3)..."""

        if self.pulse_index == None:  # If pulse hasn't fired before, this will configure the initial pulse and store it.
            if self.pulse_norm == None:  # If no custom pulse has been defined
                index = np.arange(self.size, step=self.shape[1])
                index = index[self.cell_grid[index]]  # This might cause problems if it adjusts original pulse cells... need to check.
                self.cell_grid[index] = False
            elif self.pulse_norm == True:  # If custom pulse has been defined
                index = np.ravel_multi_index(self.pulse_vectors, self.shape)
                index = index[self.cell_grid[index]]
                self.cell_grid[index] = False
            else:
                index1 = np.ravel_multi_index(self.pulse_vectors1, self.shape)
                index1 = index1[self.cell_grid[index1]]
                self.cell_grid[index1] = False
                index2 = np.ravel_multi_index(self.pulse_vectors2, self.shape)
                index2 = index2[self.cell_grid[index2]]
                self.cell_grid[index2] = False
                self.pulse_index1 = index1
                self.pulse_index2 = index2
                index = np.concatenate([index1,index2])

            self.pulse_index = index  # Variable under which pulse indices are stored
            self.exc_total.append(index)  # Appended to list of excited grid cells.
        else:  # Fires pulse indices using stored list
            index = self.pulse_index[self.cell_grid[self.pulse_index]]
            self.cell_grid[index] = False

        if len(self.excited) < self.rp:
            self.excited.append(index)
        else:
            self.excited[self.t % self.rp] = index

    def prop_tool(self, ind_list):
        # Solely used as part of Heart.propagate() to process signal propagation
        exc = []
        self.i = ind_list
        for ind in ind_list:
            ind = ind[self.cell_grid[ind]]  # Removes cells which are refractory
            if len(ind) != 0:
                norm = ind[self.cell_norm[ind]]  # Non-dysfunctional cell indices
                self.cell_grid[norm] = False  # Non-dysfunctional cells excited
                dys = ind[self.cell_dys[ind]]  # Dysfunctional cell indices
                if len(dys) != 0:
                    rand = np.random.random(len(
                        dys))  # List of random numbers between 0 and 1 for comparison to failed firing rate self.__e
                    dys_fire = dys[rand > self.__e]  # Indices of dys which do fire
                    self.cell_grid[dys_fire] = False  # Excite dys cells
                else:
                    dys_fire = np.array([], dtype='uint32')
                exc += [norm, dys_fire]
        try:
            return np.concatenate(exc)
        except:
            return np.array([], dtype='uint32')  # Important to ensure no irregularities in datatype

    def propagate(self, t_steps=1, ecg=False):

        if self.t == 0 and len(self.exc_total) == 0:
            Heart.pulse(self)

        for i in range(t_steps):
            exc_index = self.t % self.rp  # Defines current index for position in list of list of excited cells
            app_index = (self.t + 1) % self.rp
            ind = self.excited[exc_index]
            if len(ind) == 0 and self.pulse_rate == 0:
                print(self.t)
                raise ValueError(
                    'No excited cells to propagate.')  # Error only raised if there are no excited cells and a future pulse will not excite any cells.
            if self.t >= self.rp - 1:
                self.cell_grid[self.excited[app_index]] = True  # Refractory counter for all cells currently in excited list

            if len(ind) != 0:

                ind_right = ind[self.r_true[ind]] + 1
                ind_left = ind[self.l_true[ind]] -1

                ind_up = ind + self.u[ind]
                ind_down = ind + self.d[ind]
                ind_up = ind_up[self.cell_vert[ind]]  # Checks whether initial excited cell has vert connection.
                ind_down = ind_down[self.cell_vert[ind_down]]  # Checks whether below cell has vert connection.

                exc = Heart.prop_tool(self, [ind_left, ind_right, ind_up, ind_down])
            else:
                exc = np.array([], dtype='uint32')

            self.t += 1
            try:
                if self.pulse_norm:
                    if self.t % self.pulse_rate == 0:
                        index = self.pulse_index[self.cell_grid[self.pulse_index]]
                        self.cell_grid[index] = False
                        exc = np.concatenate([exc, index])
                else:
                    if self.t % self.pulse_rate1 == 0:
                        index = self.pulse_index1[self.cell_grid[self.pulse_index1]]
                        self.cell_grid[index] = False
                        exc = np.concatenate([exc, index])
                    if self.t % self.pulse_rate2 == 0:
                        index = self.pulse_index2[self.cell_grid[self.pulse_index2]]
                        self.cell_grid[index] = False
                        exc = np.concatenate([exc, index])
            except:
                pass

            if len(self.excited) < self.rp:  # Append process for list of last refractory period worth of excitations
                self.excited.append(exc)
            else:
                self.excited[app_index] = exc

            if not ecg:
                self.exc_total.append(exc)  # List containing all previously excited states

                if self.fakedata:
                    if i == t_steps - 1:
                        return self.exc_total

        if ecg:
            return exc
