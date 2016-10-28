"""
Decide on the delta/nu ranges before running this. Will get a series of questions stating parameters.
Right now parameters such as system size and rp are not currently saved so make a note of them.

You will need to install h5py. A good package which has a lot of useful modules for data science is anaconda.
"""

import basic_propagate as bp
import numpy as np
import h5py
import time


print '\n'

"""
need to add binary search.
"""

"""
Right now, need to enter ranges manually for both delta_range and nu_range
"""
delta_range = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
nu_range = np.array([0.08, 0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22])#np.arange(0.10,0.13,0.01)

eps = float(raw_input("Epsilon: "))
rp = int(raw_input("Refractory Period: "))
pulse_rate = int(raw_input("Pulse Rate: "))
sim_size = int(raw_input("Time Steps: "))
iteration_N = int(raw_input("Number of iterations: "))

print'\n'
file_name = raw_input("Name of output file: ")

h5f_par = h5py.File('%s_para.h5' % file_name, 'w')
par_grp = h5f_par.create_group('parameters')
par_grp.create_dataset('delta', data=delta_range)
par_grp.create_dataset('nu', data=nu_range)
par_grp.create_dataset('epsilon', data=eps)
par_grp.create_dataset('rp', data=rp)
par_grp.create_dataset('pulse_rate', data=pulse_rate)

h5f = h5py.File('%s.h5' % file_name, 'w')
start_time1 = time.time()
for delta in delta_range:
    grp = h5f.create_group('delta: %s' % delta)
    print "delta: %s" % delta
    for nu in nu_range:
        s_grp = grp.create_group('Nu: %s' % nu)
        print "nu: %s" % nu
        for i in range(iteration_N):
            a = bp.Heart(nu, delta, eps, rp)
            a.set_pulse(pulse_rate)
            start_time2 = time.time()
            a.propagate(sim_size)
            print("--- Iteration %s: %s seconds ---" % (i, time.time() - start_time2))
            s_grp.create_dataset('data_set_%s' % i, data=a.lenexc)
        print'\n'

print("--- Simulation: %s seconds ---" % (time.time() - start_time1))
