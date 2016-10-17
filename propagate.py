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


class Heart:

    def __init__(self, nu=0.13, delta=0.05, eps=0.05, rp=50,seed_file=None):
        """Fraction of vertical connections given: \'nu\'.
            Vertical connections are randomly filled.
            Fraction of dystfunctional cells: \'delta\'.
            Probability of failed firing: \'eps\'."""

        self.initial_seed = seed_file
        self.destroyed = []

        if self.initial_seed is None:

            self.pulse_vectors = None
            self.pulse_index = None
            self.pulse_rate = 0
            self.t = 0
            self.__n = nu  # Private vertical fractions variable
            self.__d = delta  # Private cell dysfunction variable
            self.__e = eps  # Private cell depolarisation failure variable
            self.shape = (200, 200)
            self.size = self.shape[0] * self.shape[1]
            self.__rp = rp
            self.excited = []
            self.exc_total = []
            self.state_history = {}
            self.starting_t = 0

            self.cell_grid = np.zeros(self.size,
                                      dtype='int8')  # Grid on which signal will propagate. Defines whether cell is at rest, excited or refractory.
            self.cell_vert = np.zeros(self.size,
                                      dtype='int8')  # Defines whether cell has vertical connection. 1 = Yes, 0 = No.
            self.cell_dys = np.zeros(self.size, dtype='int8')  # Defines whether cell is dysfunctional. 1 = Yes, 0 = No.
            # The above change from self.cell_type to splitting between dys and vert was necessary for the np.argwhere logic statements later.

            for i in range(self.size):
                rand_nu = np.random.random(1)[0]
                rand_delta = np.random.random(1)[0]

                if rand_nu < self.__n:  # If rand_nu < self.__n, cell (x,y) has connection to (x,y+1)
                    if rand_delta < self.__d:  # If rand_delta < self.__d, cell (x,y) is dyfunctional. Failes to fire with P = self.__e.
                        self.cell_vert[i] = 1  # Both vertically connected and dysfunctional.
                        self.cell_dys[i] = 1
                    else:
                        self.cell_vert[i] = 1  # Vertically connected but not dysfunctional.
                else:
                    if rand_delta < self.__d:  # Dysfunctional but not vertically connected.
                        self.cell_dys[i] = 1

            self.state_history[0] = np.random.get_state()

        else:

            origin = np.load("%s.npy" % self.initial_seed)

            print "Length of original simulation: %s" % len(origin[0])
            print "Please specify frame to start simulation from: (save rate: %s)" % origin[2]
            seed_frame = int(raw_input())

            self.shape = origin[1]
            self.size = self.shape[0] * self.shape[1]
            self.__rp = origin[2]
            self.__n = origin[3]  # Private vertical fractions variable
            self.__d = origin[4]  # Private cell dysfunction variable
            self.__e = origin[5]  # Private cell depolarisation failure variable
            self.state_history = origin[6]
            self.file_data = origin[0]
            self.exc_total = []  # should append the 50 excited states in here before the seed recording.

            self.initial_grid = [0] * self.size
            self.cell_vert = origin[7]
            self.cell_dys = origin[8]
            self.pulse_rate = origin[11]
            self.pulse_history = origin[12]
            self.starting_t = seed_frame
            self.pulse_vectors = self.pulse_history[1]
            self.pulse_index = self.pulse_history[0]


           # self.exc_total = origin[0][
           #                  seed_frame - self.__rp:seed_frame + 1]  # should append the 50 excited states in here before the seed recording.
            self.t = seed_frame
            np.random.set_state(self.state_history[self.t])

            excitation_level = 0
            for setup_cells in self.file_data[self.t - self.__rp + 1:self.t + 1]:
                excitation_level += 1
                for individual_cells in setup_cells.tolist():
                    self.initial_grid[individual_cells] = excitation_level

            self.cell_grid = np.array(self.initial_grid)

            excited_marker = self.t % self.__rp     # This is currently broken (compare excited cell lists for both.)
            self.excited = origin[0][seed_frame:seed_frame+1] + origin[0][seed_frame - self.__rp + 1:seed_frame - excited_marker]

    def destroy_cells(self, type, vectors_custom=None):  # Could set grid values to -1 to speed up propagate loop
        """Input vector of cells to be permanently blocked. Format as list of two lists:
        with y coordinates in list 1 and x coordinates in list 2. x = column, y = row.

        i.e. vectors = [[y1,y2,y3...],[x1,x2,x3...]]

        This will permanently block cells (x1,y1),(x2,y2),(x3,y3)..."""

        if str(type) == "custom":
            vectors = vectors_custom

        if str(type) == "square":
            print "Please input square ablation parameters:"
            print "Enter Position of square (bottom left corner)"
            position = tuple(int(x.strip()) for x in raw_input().split(','))
            print "x length:"
            y_len = int(raw_input())  # Need to flip to get desired effect
            print "y length"
            x_len = int(raw_input())  # Need to flip to get desired effect
            vectors = square_ablation(position, x_len, y_len)

        index = np.ravel_multi_index(vectors, self.shape)
        self.cell_vert[[index]] = 2
        self.cell_dys[[index]] = 2  # Permanently blocked cell
        self.destroyed[self.t] = index

    def set_pulse(self, rate, vectors=None):
        # Use before self.pulse. Defines the rate at which the pulse fires and if desired
        # defines custom pacemaker cells. If vectors = None, left most wall of tissue will be pacemaker.
        if vectors != None:
            self.pulse_vectors = vectors
        self.pulse_rate = rate

    def pulse(self):  # Still need to include functionality to avoid exciteing blocked cells
        """If cells = None, column x = 0 will be by defaults excited.
            To set custom cells to excite inputs cells as list of two lists
            with first list as y coordinates and second list as x coordinates.

            i.e. vectors = [[y1,y2,y3...],[x1,x2,x3...]]

            This will excite cells (x1,y1),(x2,y2),(x3,y3)..."""

        if self.pulse_index == None:  # If pulse hasn't fired before, this will configure the initial pulse and store it.
            if self.pulse_vectors == None:  # If no custom pulse has been defined
                index = np.arange(self.size, step=self.shape[1])
                index = index[[np.argwhere(self.cell_grid[[
                    index]] == 0).flatten()]]  # This might cause problems if it adjusts original pulse cells... need to check.
                self.cell_grid[[index]] = self.__rp
            else:  # If custom pulse has been defined
                index = np.ravel_multi_index(self.pulse_vectors, self.shape)
                index = index[[np.argwhere(self.cell_grid[[index]] == 0).flatten()]]
                self.cell_grid[[index]] = self.__rp
            self.pulse_index = index  # Variable under which pulse indices are stored
            self.exc_total.append(index)  # Appended to list of excited grid cells.
        else:  # Fires pulse indices using stored list
            index = self.pulse_index[[np.argwhere(self.cell_grid[[self.pulse_index]] == 0)]]
            self.cell_grid[[index]] = self.__rp

        if len(self.excited) < self.__rp:
            self.excited.append(index)
        else:
            self.excited[self.t % self.__rp] = index

    def prop_tool(self, ind_list):
        # Solely used as part of Heart.propagate() to process signal propagation
        exc = []
        for ind in ind_list:
            ind = ind[self.cell_grid[ind] == 0]  # Removes cells which are refractory
            if len(ind) != 0:
                norm = ind[self.cell_dys[ind] == 0]  # Non-dysfunctional cell indices
                self.cell_grid[norm] = self.__rp  # Non-dysfunctional cells excited
                dys = ind[self.cell_dys[ind] == 1]  # Dysfunctional cell indices
                if len(dys) != 0:
                    rand = np.random.random(len(
                        dys))  # List of random numbers between 0 and 1 for comparison to failed firing rate self.__e
                    dys_fire = dys[rand > self.__e]  # Indices of dys which do fire
                    self.cell_grid[dys_fire] = self.__rp  # Excite dys cells
                else:
                    dys_fire = np.array([], dtype='int32')
                exc += [norm, dys_fire]
        try:
            return np.concatenate(exc)
        except:
            return np.array([], dtype='int32')  # Important to ensure no irregularities in datatype

    def propagate(self, t_steps=1):
        if self.t == 0 and len(self.exc_total) == 0:
            Heart.pulse(self)

        for i in range(t_steps):
            exc_index = self.t % self.__rp  # Defines current index for position in list of list of excited cells
            if len(self.excited[exc_index]) == 0 and self.pulse_rate == 0:
                print self.t
                raise ValueError(
                    'No excited cells to propagate.')  # Error only raised if there are no excited cells and a future pulse will not excite any cells.
            ind = self.excited[exc_index]
            self.cell_grid[
                np.concatenate(self.excited)] -= 1  # Refractory counter for all cells currently in excited list

            if len(self.excited[exc_index]) != 0:

                # print len(self.excited)
                ind_up = ind + self.shape[0]  # Index of cells directly above initially excited cells
                ind_down = ind - self.shape[0]  # Index of cells below
                ind_up[ind_up >= self.size] -= self.size
                ind_down[ind_down < 0] += self.size

                ind_right = ind[
                    ind % self.shape[1] != self.shape[1] - 1]  # Deletes any cells outside of right tissue boundary
                ind_right += 1  # Above: the entries corresponding to the indices of 'ind' where the remainder when dividing by grid length is not grid length - 1
                ind_left = ind[ind % self.shape[1] != 0]  # Deletes any cells outside of left tissue boundary
                ind_left -= 1  # Above: the entries corresponding to the indices of 'ind' where the remainder when dividing by grid length is not 0

                ind_up = ind_up[self.cell_vert[ind] == 1]  # Checks whether initial excited cell has vert connection.
                ind_down = ind_down[self.cell_vert[ind_down] == 1]  # Checks whether below cell has vert connection.

                exc = Heart.prop_tool(self, [ind_left, ind_right, ind_up, ind_down])
            else:
                exc = np.array([], dtype='int32')

            self.t += 1  # next time step
            app_index = self.t % self.__rp  # index of self.excited which should be replaced by current temporary list

            try:
                if self.t % self.pulse_rate == 0:  # If time is multiple of pulse rate, pulse cells fire
                    print self.t
                    index = self.pulse_index[self.cell_grid[self.pulse_index] == 0]
                    index = index[self.cell_dys[index] != 2]  # Does not fire dead cells
                    self.cell_grid[index] = self.__rp
                    exc = np.concatenate([exc, index])
            except:
                pass

            if len(self.excited) < self.__rp:  # Append process for list of last refractory period worth of excitations
                self.excited.append(exc)
            else:
                self.excited[app_index] = exc

            if self.t % self.__rp == 0:
                self.state_history[self.t] = np.random.get_state()  # Seed recording for generator.

            self.pulse_history = (self.pulse_index, self.pulse_vectors)
            self.exc_total.append(exc)  # List containing all previously excited states

    def save(self, file_name):
        # pickle.dump((self.exc_total,self.shape,self.__rp), open("%s.p" % file_name, 'wb'))
        np.save(str(file_name), (self.exc_total, self.shape, self.__rp,
                                 self.__n, self.__d, self.__e, self.state_history,
                                 self.cell_vert, self.cell_dys, self.destroyed, self.starting_t,
                                 self.pulse_rate, self.pulse_history))
