"""
Streamlined propagate for use with simulate.py
Only returns len(exc)

----> Need to check that its still giving the correct data with changes made <-------
"""

import numpy as np


def square_ablation(position, x_len, y_len):
    x_index = []
    y_index = []
    y_ref, x_ref = position  # Needs to be flipped to get desired effect. y_ref <--> x_ref
    for i in range(x_ref, x_ref + x_len):
        for j in range(y_ref, y_ref + y_len):
            x_index.append(i)
            y_index.append(j)
    return [x_index, y_index]


def fake_af(): #Returns idealised AF inducing heart with single critical circuit initiated by a single dysfunctional cell.
    a = Heart(1,0.,0.2,50)
    a.cell_vert[20080:20120] = False
    a.cell_vert[20280:20320] = False
    a.cell_dys[20280] = True
    a.cell_norm = np.invert(a.cell_dys)
    a.set_pulse(220)
    return a


class Heart:

    def __init__(self, nu=0.5, delta=0.05, eps=0.05, rp=50, x_size=200, y_size=200, fakedata=False):
        """Fraction of vertical connections given: \'nu\'.
            Vertical connections are randomly filled.
            Fraction of dysfunctional cells: \'delta\'.
            Probability of failed firing: \'eps\'.

            count_excited used by analysis.
            To break at AF initiation set count_excited = \'start\'
            To count time spent in AF set count_excited = \'time\'"""

        self.destroyed = {}
        self.pulse_vectors = None
        self.pulse_index = None
        self.pulse_history = None
        self.cell_alive = None
        self.pulse_rate = 0
        self.t = 0
        self.nu = nu  # Private vertical fractions variable
        self.delta = delta  # Private cell dysfunction variable
        self.eps = eps  # Private cell depolarisation failure variable
        self.shape = (y_size, x_size)
        self.size = x_size * y_size
        self.rp = rp
        self.excited = []
        self.exc_total = []
        self.r_true = (np.arange(self.size) % self.shape[1] != self.shape[1] - 1)
        self.l_true = (np.arange(self.size) % self.shape[1] != 0)
        self.u = np.ones(self.size, dtype='int32') * self.shape[0]
        self.u[-self.shape[1]::] = - self.size + self.shape[0]
        self.d = np.ones(self.size, dtype='int32') * - self.shape[0]
        self.d[:self.shape[1]:] = + self.size - self.shape[0]

        # Grid on which signal will propagate.
        # Defines whether cell is at rest, excited or refractory.
        self.cell_grid = np.ones(self.size,
                                 dtype='bool')
        # Defines whether cell has vertical connection. 1 = Yes, 0 = No.
        self.cell_vert = np.zeros(self.size,
                                  dtype='bool')
        # Defines whether cell is dysfunctional. 1 = Yes, 0 = No.
        self.cell_dys = np.zeros(self.size, dtype='bool')

        # The above change from self.cell_type to splitting between dys and vert was necessary
        # for the np.argwhere logic statements later.

        if not fakedata:
            """
            Regular propagate.
            """
            self.starting_t = np.empty(0, dtype='uint32')
            self.cell_alive = np.ones(self.size, dtype='bool')
            self.any_ablate = False
            self.lenexc = None
            self.fakedata = False

            for i in range(self.size):
                rand_nu = np.random.random(1)[0]
                rand_delta = np.random.random(1)[0]

                # If rand_nu < self.__n, cell (x,y) has connection to (x,y+1)
                if rand_nu < self.nu:
                    # If rand_delta < self.__d, cell (x,y) is dyfunctional. Failes to fire with P = self.eps.
                    if rand_delta < self.delta:
                        self.cell_vert[i] = True  # Both vertically connected and dysfunctional.
                        self.cell_dys[i] = True
                    else:
                        self.cell_vert[i] = True  # Vertically connected but not dysfunctional.
                else:
                    if rand_delta < self.delta:  # Dysfunctional but not vertically connected.
                        self.cell_dys[i] = True

            self.cell_norm = np.invert(self.cell_dys)

        if fakedata:
            """
            Ideal AF is formed in this propagate.
            """
            self.fakedata = fakedata
            x = np.random.randint(40000)
            while x % 200 > 160:
                x = np.random.randint(40000)
            if x < 39800:
                y = x + 200
            else:
                y = (x + 200) % 40000
            self.x = x
            self.cell_vert[x:x + 30] = False
            self.cell_vert[y:y + 30] = False
            self.cell_dys[y] = True
            self.cell_norm = np.invert(self.cell_dys)

    def destroy_cells(self, type, vectors_custom=None):  # Could set grid values to -1 to speed up propagate loop
        """Input vector of cells to be permanently blocked. Format as list of two lists:
        with y coordinates in list 1 and x coordinates in list 2. x = column, y = row.

        i.e. vectors = [[y1,y2,y3...],[x1,x2,x3...]]

        This will permanently block cells (x1,y1),(x2,y2),(x3,y3)..."""

        if str(type) == "custom":
            vectors = vectors_custom

        if str(type) == "square":
            print(("Please input square ablation parameters:"))
            print(("Enter Position of square (tuple) (bottom left corner)"))
            position = tuple(int(x.strip()) for x in input().split(','))
            print(("x length:"))
            y_len = int(input())  # Need to flip to get desired effect
            print(("y length"))
            x_len = int(input())  # Need to flip to get desired effect
            vectors = square_ablation(position, x_len, y_len)

        if str(type) == "chevron":
            print( "Please input square ablation parameters:")
            print( "Chevron tip pointing to left (l) or right (r)?")
            direction = str(input())
            print( "Enter x Position of chevron tip")
            x = int(input())
            if x < 0 or x > self.shape[1] - 1:
                raise ValueError('x not in tissue range')
            print( "Enter y Position of chevron tip")
            y = int(input())
            if y < 0 or y > self.shape[0] - 1:
                raise ValueError('y not in tissue range')
            print( "chevron length:")
            chev_len = int(input())  # Need to flip to get desired effect

            ind = int(x + (y * self.shape[0]))
            u = ind
            d = ind
            index = [ind]
            if direction == 'r':
                a = -1
            if direction == 'l':
                a = 1
            for i in range(chev_len - 1):
                u += (a + self.shape[0])
                d += (a - self.shape[0])
                if u % self.shape[1] == 0 and a == 1:
                    break
                if u % self.shape[1] == self.shape[1] - 1 and a == -1:
                    break
                index.append(u)
                index.append(d)
            index = np.array(index)
            for i in range(len(index)):
                if index[i] < 0:
                    index[i] += self.size
                if index[i] >= self.size:
                    index[i] -= self.size

        if str(type) != "chevron":
            index = np.ravel_multi_index(vectors, self.shape)
        self.cell_alive[index] = False
        self.any_ablate = True
        self.destroyed.setdefault(self.t, [])  # For when multiple ablations happen at the same time.
        self.destroyed[self.t].append(index)

    def set_pulse(self, rate, vectors=None):
        # Use before self.pulse. Defines the rate at which the pulse fires and if desired
        # defines custom pacemaker cells. If vectors = None, left most wall of tissue will be pacemaker.
        if vectors is not None:
            self.pulse_vectors = vectors
        self.pulse_rate = rate

    def pulse(self):  # Still need to include functionality to avoid exciteing blocked cells

        """If cells = None, column x = 0 will be by defaults excited.
            To set custom cells to excite inputs cells as list of two lists
            with first list as y coordinates and second list as x coordinates.

            i.e. vectors = [[y1,y2,y3...],[x1,x2,x3...]]

            This will excite cells (x1,y1),(x2,y2),(x3,y3)..."""
        if self.cell_alive is None:

            # If pulse hasn't fired before, this will configure the initial pulse and store it.
            if self.pulse_index is None:
                if self.pulse_vectors is None:  # If no custom pulse has been defined
                    index = np.arange(self.size, step=self.shape[1])
                    # This might cause problems if it adjusts original pulse cells... need to check.
                    index = index[self.cell_grid[index]]
                    self.cell_grid[index] = False
                else:  # If custom pulse has been defined
                    index = np.ravel_multi_index(self.pulse_vectors, self.shape)
                    index = index[self.cell_grid[index]]
                    self.cell_grid[index] = False
                self.pulse_index = index  # Variable under which pulse indices are stored
                self.exc_total.append(index)  # Appended to list of excited grid cells.
            else:  # Fires pulse indices using stored list
                index = self.pulse_index[self.cell_grid[self.pulse_index]]
                self.cell_grid[index] = False

            if len(self.excited) < self.rp:
                self.excited.append(index)
            else:
                self.excited[self.t % self.rp] = index

        else:
            # If pulse hasn't fired before, this will configure the initial pulse and store it.
            if self.pulse_index == None:
                if self.pulse_vectors == None:  # If no custom pulse has been defined
                    index = np.arange(self.size, step=self.shape[1])
                    index = index[self.cell_alive[index]]
                    # This might cause problems if it adjusts original pulse cells... need to check.
                    index = index[self.cell_grid[index]]
                    self.cell_grid[index] = False
                else:  # If custom pulse has been defined
                    index = np.ravel_multi_index(self.pulse_vectors, self.shape)
                    index = index[self.cell_alive[index]]
                    index = index[self.cell_grid[index]]
                    self.cell_grid[index] = False
                self.pulse_index = index  # Variable under which pulse indices are stored
                self.exc_total = self.exc_total + self.pulse_index.tolist()
                self.lenexc[0] = len(self.pulse_index)
            else:  # Fires pulse indices using stored list
                index = self.pulse_index[self.cell_grid[self.pulse_index]]
                index = index[self.cell_alive[index]]
                self.cell_grid[index] = False

            if len(self.excited) < self.rp:
                self.excited.append(index)
            else:
                self.excited[self.t % self.rp] = index

    def prop_tool(self, ind_list):
        # Solely used as part of Heart.propagate() to process signal propagation
        exc = []
        for ind in ind_list:
            ind = ind[self.cell_grid[ind]]  # Removes cells which are refractory
            if self.any_ablate:
                ind = ind[self.cell_alive[ind]]
            if len(ind) != 0:
                norm = ind[self.cell_norm[ind]]  # Non-dysfunctional cell indices
                self.cell_grid[norm] = False  # Non-dysfunctional cells excited
                dys = ind[self.cell_dys[ind]]  # Dysfunctional cell indices
                if len(dys) != 0:
                    rand = np.random.random(len(
                        dys))  # List of random numbers between 0 and 1 for comparison to failed firing rate self.__e
                    dys_fire = dys[rand > self.eps]  # Indices of dys which do fire
                    self.cell_grid[dys_fire] = False  # Excite dys cells
                else:
                    dys_fire = np.array([], dtype='uint32')
                exc += [norm, dys_fire]
        try:
            return np.concatenate(exc)
        except:
            return np.array([], dtype='uint32')  # Important to ensure no irregularities in datatype

    def propagate(self, t_steps=1, real_time=False, ecg=False, both=False, data_range=None):

        if not self.fakedata:
            temp_data_range = list()
            self.lenexc = np.zeros(t_steps + 1, dtype='uint32')
            if self.t == 0:
                Heart.pulse(self)

            for i in range(t_steps):
                exc_index = self.t % self.rp  # Defines current index for position in list of list of excited cells
                app_index = (self.t + 1) % self.rp
                ind = self.excited[exc_index]
                if len(ind) == 0 and self.pulse_rate == 0:
                    print((self.t))
                    # Error only raised if there are no excited cells and a future pulse will not excite any cells.
                    raise ValueError(
                        'No excited cells to propagate.')
                # Refractory counter for all cells currently in excited list
                if self.t >= self.rp - 1:
                    self.cell_grid[self.excited[app_index]] = True

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
                    if self.t % self.pulse_rate == 0:  # If time is multiple of pulse rate, pulse cells fire
                        index = self.pulse_index[self.cell_grid[self.pulse_index]]
                        if self.any_ablate:
                            index = index[self.cell_alive[index]]  # Does not fire dead cells
                        self.cell_grid[index] = False
                        exc = np.concatenate([exc, index])
                except:
                    pass

                # Append process for list of last refractory period worth of excitations
                if len(self.excited) < self.rp:
                    self.excited.append(exc)
                else:
                    self.excited[app_index] = exc

                if not real_time:
                    self.pulse_history = (self.pulse_index, self.pulse_vectors)
                    self.lenexc[i+1] = len(exc)

                if data_range:
                    temp_data_range.append(exc)

            if real_time:
                return len(exc)

            if ecg:
                return exc

            if both:
                return exc, len(exc)

            if data_range:
                return temp_data_range

        if self.fakedata:
            counter = 0
            if self.t == 0 and len(self.exc_total) == 0:
                Heart.pulse(self)

            while counter <= 1100:
                exc_index = self.t % self.rp  # Defines current index for position in list of list of excited cells
                app_index = (self.t + 1) % self.rp
                ind = self.excited[exc_index]
                if len(ind) == 0 and self.pulse_rate == 0:
                    print((self.t))
                    # Error only raised if there are no excited cells and a future pulse will not excite any cells.
                    raise ValueError(
                        'No excited cells to propagate.')
                if self.t >= self.rp - 1:
                    self.cell_grid[
                        self.excited[app_index]] = True  # Refractory counter for all cells currently in excited list

                if len(ind) != 0:

                    ind_right = ind[self.r_true[ind]] + 1
                    ind_left = ind[self.l_true[ind]] - 1

                    ind_up = ind + self.u[ind]
                    ind_down = ind + self.d[ind]
                    ind_up = ind_up[self.cell_vert[ind]]  # Checks whether initial excited cell has vert connection.
                    ind_down = ind_down[self.cell_vert[ind_down]]  # Checks whether below cell has vert connection.

                    exc = Heart.prop_tool(self, [ind_left, ind_right, ind_up, ind_down])
                else:
                    exc = np.array([], dtype='uint32')

                self.t += 1
                try:
                    if self.t % self.pulse_rate == 0:
                        index = self.pulse_index[self.cell_grid[self.pulse_index]]
                        self.cell_grid[index] = False
                        exc = np.concatenate([exc, index])
                except:
                    pass

                # Append process for list of last refractory period worth of excitations
                if len(
                        self.excited) < self.rp:
                    self.excited.append(exc)
                else:
                    self.excited[app_index] = exc
                self.exc_total.append(exc)  # List containing all previously excited states

                if self.fakedata:
                    if len(self.exc_total[-1]) > 220:
                        if counter == 1100:
                            return self.exc_total[-1100:-220], self.x
                        else:
                            counter += 1
                    else:
                        counter = 0
