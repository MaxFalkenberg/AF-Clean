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
import argparse


parser = argparse.ArgumentParser(description='max stuff')
parser.add_argument('iterations', metavar='ITER', type=int,
                    help='an integer for the accumulator')
parser.add_argument('nu', metavar='NU', type=float,
                                        help='an integer for the accumulator')
parser.add_argument('output', metavar='OUTPUT',
                    help ='an integer for the accumulator')
parser.add_argument('mode', metavar='MODE',
                                        help ='[r,s,g,c (have to set in code)]')

args = parser.parse_args()
print(args)
# import matplotlib.pyplot as plt

Iterations = args.iterations #int(input("Number of iterations: "))

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

e = at.ECG(shape=(200, 200), probe_height=3, m=args.mode)  # Assuming shape/probe height doesn't change.
file_name = args.output #nput("Name of output file: ")
#print("Nu Value:")
nu = args.nu#float(input())

print(file_name, nu, e.mode, Iterations)
# print(nu)

h5f = h5py.File('%s.h5' % file_name, 'w')
for index in range(Iterations):
    #nu = (np.random.rand() / 5) + 0.1
    start_time1 = time.time()
    index_grp = h5f.create_group('Index: %s' % index)

    a = fp.Heart(nu = nu, fakedata=True)
    crit_position = np.random.randint(40000)
    y_rand,x_rand = np.unravel_index(crit_position,(200,200))
    while x_rand > 169:
        crit_position = np.random.randint(40000)
        y_rand,x_rand = np.unravel_index(crit_position,(200,200))
    x_rand2 = x_rand + 30
    a.set_pulse(60,[[y_rand,y_rand],[x_rand,x_rand2]])
    # print [[y_rand,y_rand],[x_rand,x_rand2]]
    raw_data = a.propagate(1260)

    converted_data = list()
    grid = np.zeros(a.shape)
    convert(raw_data, converted_data)
    np.save('raw_data.npy',np.array(converted_data))

    # Saving the critical circuit position
    index_grp.create_dataset('Crit Position', data=crit_position)
    index_grp.create_dataset('Nu', data=nu)
    ecg = e.solve(converted_data[1111:])


    index_grp.create_dataset('ECG', data=ecg)
    index_grp.create_dataset('Probe Positions', data=e.probe_position)
    # index_grp.create_dataset('Probe Number', data=e.probe_number)
    print("--- Iteration %s: %s seconds ---" % (index, time.time() - start_time1))

h5f.close()
