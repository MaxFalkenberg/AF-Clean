import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


class Heart:

    def __init__(self,nu = 1.,delta = 0.05, eps = 0.05, rp = 50):
        """Fraction of vertical connections given: \'nu\'.
            Vertical connections are randomly filled.
            Fraction of dystfunctional cells: \'delta\'.
            Probability of failed firing: \'eps\'."""

        self.__n = nu    #Private vertical fractions variable
        self.__d = delta #Private cell dysfunction variable
        self.__e = eps   #Private cell depolarisation failure variable
        self.__t = 0
        self.excited  = []
        self.exc_total = []
        self.__rp = rp
        self.shape = (200,200)
        self.size = self.shape[0] * self.shape[1]
        self.pulse_rate = None
        self.pulse_vectors = None
        self.pulse_index = None

        self.cell_grid = np.zeros(self.size, dtype = 'int8') #Grid on which signal will propagate. Defines whether cell is at rest, excited or refractory.
        self.cell_vert = np.zeros(self.size, dtype = 'int8') #Defines whether cell has vertical connection. 1 = Yes, 0 = No.
        self.cell_dys = np.zeros(self.size, dtype = 'int8') #Defines whether cell is dysfunctional. 1 = Yes, 0 = No.
        #The above change from self.cell_type to splitting between dys and vert was necessary for the np.argwhere logic statements later.

        for i in range(self.size):
            rand_nu = random.random()
            rand_delta = random.random()

            if rand_nu < self.__n: #If rand_nu < self.__n, cell (x,y) has connection to (x,y+1)
                if rand_delta < self.__d: #If rand_delta < self.__d, cell (x,y) is dyfunctional. Failes to fire with P = self.__e.
                    self.cell_vert[i] = 1 #Both vertically connected and dysfunctional.
                    self.cell_dys[i] = 1
                else:
                    self.cell_vert[i] = 1 #Vertically connected but not dysfunctional.
            else:
                if rand_delta < self.__d: #Dysfunctional but not vertically connected.
                    self.cell_dys[i] = 1

    def destroy_cells(self,vectors): #Could set grid values to -1 to speed up propagate loop
        """Input vector of cells to be permanently blocked. Format as list of two lists:
        with y coordinates in list 1 and x coordinates in list 2. x = column, y = row.

        i.e. vectors = [[y1,y2,y3...],[x1,x2,x3...]]

        This will permanently block cells (x1,y1),(x2,y2),(x3,y3)..."""
        index = np.ravel_multi_index(vectors,self.shape)
        self.cell_vert[[index]] = 2
        self.cell_dys[[index]] = 2      #Permanently blocked cell


    def set_pulse(self,rate,vectors = None):
        #Use before self.pulse. Defines the rate at which the pulse fires and if desired
        #defines custom pacemaker cells. If vectors = None, left most wall of tissue will be pacemaker.
        if vectors != None:
            self.pulse_vectors = vectors
        self.pulse_rate = rate

    def pulse(self): #Still need to include functionality to avoid exciteing blocked cells
        """If cells = None, column x = 0 will be by defaults excited.
            To set custom cells to excite inputs cells as list of two lists
            with first list as y coordinates and second list as x coordinates.

            i.e. vectors = [[y1,y2,y3...],[x1,x2,x3...]]

            This will excite cells (x1,y1),(x2,y2),(x3,y3)..."""

        if self.pulse_index == None: #If pulse hasn't fired before, this will configure the initial pulse and store it.
            if self.pulse_vectors == None: #If no custom pulse has been defined
                index = np.arange(self.size,step = self.shape[1])
                index = index[[np.argwhere(self.cell_grid[[index]] == 0).flatten()]] #This might cause problems if it adjusts original pulse cells... need to check.
                self.cell_grid[[index]] = self.__rp
            else: #If custom pulse has been defined
                index = np.ravel_multi_index(self.pulse_vectors,self.shape)
                index = index[[np.argwhere(self.cell_grid[[index]] == 0).flatten()]]
                self.cell_grid[[index]] = self.__rp
            self.pulse_index = index #Variable under which pulse indices are stored
            self.exc_total.append(index) #Appended to list of excited grid cells.
        else: #Fires pulse indices using stored list
             index = self.pulse_index[[np.argwhere(self.cell_grid[[self.pulse_index]] == 0)]]
             self.cell_grid[[index]] = self.__rp

        if len(self.excited) < self.__rp:
            self.excited.append(index)
        else:
            self.excited[self.__t % self.__rp] = index

    def propagate(self,t_steps = 1):

        for i in range(t_steps):
            #print self.cell_grid
            exc = []
            exc_index = self.__t % self.__rp #Defines current index for position in list of list of excited cells
            if len(self.excited[exc_index]) == 0 and self.pulse_rate == 0:
                raise ValueError('No excited cells to propagate.') #Error only raised if there are no excited cells and a future pulse will not excite any cells.
            ind = self.excited[exc_index]

            for j in self.excited:
                if len(j) != 0:
                    self.cell_grid[[j]] -= 1 #Refractory counter for all cells currently in excited list

            #print len(self.excited)
            ind_up = ind + self.shape[0] #Index of cells directly above initially excited cells
            ind_down = ind - self.shape[0] #Index of cells below
            for k in range(len(ind_up)): #Enforces continuous boundary conditions on top
                if ind_up[k] >= self.size:
                    ind_up[k] -= self.size
            for k in range(len(ind_down)):#Enforces continuous boundary conditions on bottom
                if ind_down[k] < 0:
                    ind_down[k] += self.size
            ind_right = ind[[np.argwhere(ind % self.shape[1] != self.shape[1] - 1).flatten()]] #Deletes any cells outside of right tissue boundary
            ind_right += 1 #Above: the entries corresponding to the indices of 'ind' where the remainder when dividing by grid length is not grid length - 1
            ind_left = ind[[np.argwhere(ind % self.shape[1] != 0).flatten()]] #Deletes any cells outside of left tissue boundary
            ind_left -= 1 #Above: the entries corresponding to the indices of 'ind' where the remainder when dividing by grid length is not 0

            ind_left = ind_left[[np.argwhere(self.cell_grid[[ind_left]] == 0).flatten()]] #Removes cells which are refractory
            if len(ind_left) != 0:
                norm = ind_left[[np.argwhere(self.cell_dys[[ind_left]] == 0).flatten()]] #Non-dysfunctional cell indices
                #print 1, self.cell_grid[[norm]]
                self.cell_grid[[norm]] = self.__rp #Non-dysfunctional cells excited
                #print 1, norm
                exc += norm.tolist() #Add excited cells to temporary list of cells excited at this time step.
                dys = ind_left[[np.argwhere(self.cell_dys[[ind_left]] == 1).flatten()]] #Dysfunctional cell indices
                rand = np.random.random(len(dys)) #List of random numbers between 0 and 1 for comparison to failed firing rate self.__e
                dys_fire = dys[[np.argwhere(rand > self.__e).flatten()]] #Indices of dys which do fire
                #print 2, self.cell_grid[[dys_fire]]
                self.cell_grid[[dys_fire]] = self.__rp #Excite dys cells
                exc += dys_fire.tolist() #Add excited cells to temporary list of cells excited at this time step.
                #print 2, dys_fire

            ind_right = ind_right[[np.argwhere(self.cell_grid[[ind_right]] == 0).flatten()]] #same as ind_left for cells on right of excited.
            if len(ind_right) != 0:
                norm = ind_right[[np.argwhere(self.cell_dys[[ind_right]] == 0).flatten()]]
                #print 3, self.cell_grid[[norm]]
                self.cell_grid[[norm]] = self.__rp
                #print 3, norm
                exc += norm.tolist()
                dys = ind_right[[np.argwhere(self.cell_dys[[ind_right]] == 1).flatten()]]
                rand = np.random.random(len(dys))
                dys_fire = dys[[np.argwhere(rand > self.__e).flatten()]]
                #print 4, self.cell_grid[[dys_fire]]
                self.cell_grid[[dys_fire]] = self.__rp
                exc += dys_fire.tolist()
                #print 4, dys_fire
            self.ind_up = ind_up

            ind_up = ind_up[[np.argwhere(self.cell_vert[[ind]] == 1).flatten()]] #Same as ind_left for cells above. Checks whether initial excited cell has vert connection.
            ind_up = ind_up[[np.argwhere(self.cell_grid[[ind_up]] == 0).flatten()]]
            if len(ind_up) != 0:

                norm = ind_up[[np.argwhere(self.cell_dys[[ind_up]] == 0).flatten()]]
                #print 5, self.cell_grid[[norm]]
                self.cell_grid[[norm]] = self.__rp
                exc += norm.tolist()
                dys = ind_up[[np.argwhere(self.cell_dys[[ind_up]] == 1).flatten()]]
                rand = np.random.random(len(dys))
                dys_fire = dys[[np.argwhere(rand > self.__e).flatten()]]
                #print 6, self.cell_grid[[dys_fire]]
                self.cell_grid[[dys_fire]] = self.__rp
                exc += dys_fire.tolist()

            ind_down = ind_down[[np.argwhere(self.cell_vert[[ind_down]] == 1).flatten()]]#Same as ind_left for cells below. Checks whether below cell has vert connection.
            ind_down = ind_down[[np.argwhere(self.cell_grid[[ind_down]] == 0).flatten()]]
            if len(ind_down) != 0:

                norm = ind_down[[np.argwhere(self.cell_dys[[ind_down]] == 0).flatten()]]
                #print 7, self.cell_grid[[norm]]
                self.cell_grid[[norm]] = self.__rp
                #print 7, norm
                exc += norm.tolist()
                dys = ind_down[[np.argwhere(self.cell_dys[[ind_down]] == 1).flatten()]]
                rand = np.random.random(len(dys))
                dys_fire = dys[[np.argwhere(rand > self.__e).flatten()]]
                #print 8, self.cell_grid[[dys_fire]]
                self.cell_grid[[dys_fire]] = self.__rp
                exc += dys_fire.tolist()
                #print 8, dys_fire
        self.__t += 1 #next time step
        app_index = self.__t % self.__rp #index of self.excited which should be replaced by current temporary list

        if self.__t % self.pulse_rate == 0: #If time is multiple of pulse rate, pulse cells fire
            print self.__t
            index = self.pulse_index[[np.argwhere(self.cell_grid[[self.pulse_index]] == 0).flatten()]].tolist()
            self.cell_grid[[index]] = self.__rp
            exc += index

        if len(self.excited) < self.__rp: #Append process for list of last refractory period worth of excitations
            self.excited.append(np.array(exc, dtype = 'int32'))
        else:
            self.excited[app_index] = np.array(exc, dtype = 'int32')

        self.exc_total.append(np.array(exc, dtype = 'int32')) #List containing all previously excited states

    def save(self, file_name):

        pickle.dump((self.exc_total,self.shape,self.__rp), open("%s.p" % file_name, 'wb'))
