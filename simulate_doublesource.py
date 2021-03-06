"""
Decide on the delta/nu ranges before running this. Will get a series of questions stating parameters.
Right now parameters such as system size and rp are not currently saved so make a note of them.

You will need to install h5py. A good package which has a lot of useful modules for data science is anaconda.
"""

import propagate_singlesource as fp
import analysis_theano as at
from itertools import product
import numpy as np
import h5py
import time
# import matplotlib.pyplot as plt


Iterations = int(input("Number of iterations: "))

def convert(data, output):

    for index_data in data:
        grid[(grid > 0) & (grid <= 50)] -= 1
        if index_data == []:  # could use <if not individual_data.any():> but this is more readable.
            current_state = grid.copy()
            output.append(current_state)
        else:
            indices = np.unravel_index(index_data, a.shape)
            for ind in range(len(indices[0])):
                grid[indices[0][ind]][indices[1][ind]] = 50
            current_state = grid.copy()
            output.append(current_state)

    return output

e = at.ECG(shape=(200, 200), probe_height=3)  # Assuming shape/probe height doesn't change.
file_name = input("Name of output file: ")
print("Nu Value:")
nu = float(input())

h5f = h5py.File('%s.h5' % file_name, 'w')
for index in range(Iterations):
    start_time1 = time.time()
    index_grp = h5f.create_group('Index: %s' % index)

    a = fp.Heart(nu = nu, fakedata=True)
    rand_true = True
    while rand_true:
        crit_position = np.random.randint(0,40000,2)
        y_rand,x_rand = np.unravel_index(crit_position,(200,200))
        if (((y_rand[1] - y_rand[0]) ** 2) + ((x_rand[1] - x_rand[0]) ** 2)) ** 0.5 > 50.:
            rand_true = False

    a.set_pulse(60,[[y_rand],[x_rand]])
    raw_data = a.propagate(960)
    converted_data = list()
    grid = np.zeros(a.shape)
    convert(raw_data, converted_data)

    # Saving the critical circuit position
    index_grp.create_dataset('Crit Position', data=crit_position)
    ecg = e.solve(converted_data[661:])

    index_grp.create_dataset('ECG', data=ecg)
    index_grp.create_dataset('Probe Positions', data=e.probe_position)
    print("--- Iteration %s: %s seconds ---" % (index, time.time() - start_time1))

h5f.close()
