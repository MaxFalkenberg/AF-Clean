"""
Decide on the delta/nu ranges before running this. Will get a series of questions stating parameters.
Right now parameters such as system size and rp are not currently saved so make a note of them.

You will need to install h5py. A good package which has a lot of useful modules for data science is anaconda.

Now has ML-Train option to generate ML training data.

Sampling has been added to explore long simulations of 1000*1000 grids.
"""

import basic_propagate as bp
import propagate_fakedata as fp
import analysis_theano as at
import numpy as np
import h5py
import time
# import matplotlib.pyplot as plt

print '[Delta, ML-Train, Sampling]'

Simulation_type = raw_input("Please the simulation type: ")

if Simulation_type == 'Delta':

    """
    need to add binary search.
    Right now, need to enter ranges manually for both delta_range and nu_range
    """

    delta_range = np.array([0.05])
    nu_range = np.array([0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22])

    print "Delta: %s" % delta_range
    print "Nu: %s" % nu_range

    eps = float(raw_input("Epsilon: "))
    rp = int(raw_input("Refractory Period: "))
    pulse_rate = int(raw_input("Pulse Rate: "))
    sim_size = int(raw_input("Time Steps: "))
    iteration_N = int(raw_input("Number of iterations: "))

    print'\n'
    file_name = raw_input("Name of output file: ")

    h5f_par = h5py.File('%s_para.h5' % file_name, 'w')
    par_grp = h5f_par.create_group('parameters')
    par_grp.create_dataset('Delta', data=delta_range)
    par_grp.create_dataset('Nu', data=nu_range)
    par_grp.create_dataset('Epsilon', data=eps)
    par_grp.create_dataset('Refractory Period', data=rp)
    par_grp.create_dataset('Pulse Rate', data=pulse_rate)
    par_grp.create_dataset('Simulation Size', data=sim_size)
    par_grp.create_dataset('Iterations', data=iteration_N)

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
                a.propagate(sim_size, real_time=False, ecg=False)
                print("--- Iteration %s: %s seconds ---" % (i, time.time() - start_time2))
                s_grp.create_dataset('data_set_%s' % i, data=a.lenexc)
            print'\n'

    print("--- Simulation: %s seconds ---" % (time.time() - start_time1))
    h5f.close()

if Simulation_type == 'ML-Train':

    print "Creating training data from propagate_fakedata.py"
    Iterations = int(raw_input("Number of iterations: "))

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
    file_name = raw_input("Name of output file: ")

    if e.mode == 'range':

        h5f = h5py.File('%s.h5' % file_name, 'w')
        for index in range(Iterations):
            start_time1 = time.time()
            # Group Creation
            index_grp = h5f.create_group('Index: %s' % index)
            # Subgroups creation
            # probe_sgrp = index_grp.create_group('Probe position')
            # ecg_sgrp = index_grp.create_group('ECG')

            a = fp.Heart(fakedata=True)
            raw_data, crit_position = a.propagate()
            print crit_position
            converted_data = list()
            grid = np.zeros(a.shape)
            convert(raw_data, converted_data)
            print converted_data

            # Saving the critical circuit position
            index_grp.create_dataset('Crit Position', data=crit_position)

            ecg = e.solve(converted_data)

            index_grp.create_dataset('ECG', data=ecg)
            index_grp.create_dataset('Probe Positions', data=e.probe_position)
            print("--- Iteration %s: %s seconds ---" % (index, time.time() - start_time1))

        h5f.close()

    if e.mode == 'single':

        print "Test right now. Only returns  converted data."

        a = fp.Heart(fakedata=True)
        raw_data, crit_position = a.propagate()
        converted_data = list()
        grid = np.zeros(a.shape)
        convert(raw_data, converted_data)
        print converted_data

if Simulation_type == 'Sampling':

        print "Simulation of heart tissue with defined sample rate."
        time_steps = int(raw_input("Number of time steps: "))
        # Needs to be larger than 50.
        # Should indicate frame difference between sampling.
        sample_interval = int(raw_input("Sample spacing: "))
        # sample range (has to over 50)
        sample_range = int(raw_input("Sample range: "))
        file_name = raw_input("Name of output file: ")

        h5f = h5py.File('%s.h5' % file_name, 'w')

        a = bp.Heart(nu=0.17, eps=0.05, delta=0.05, shape=(1000, 1000))
        a.set_pulse(220)

        sample_steps = range(sample_interval, time_steps, sample_interval)
        animation_grid = np.zeros(a.shape)
        print sample_steps

        upper_limit = len(sample_steps)
        count = 0

        for sample in sample_steps:

            unrecorded_jump = sample - a.rp - count
            count += unrecorded_jump
            print count
            start_time1 = time.time()
            a.propagate(unrecorded_jump)
            group = h5f.create_group('Sample: %s' % sample)
            data = a.propagate(sample_range + a.rp, data_range=True)
            for index, i in enumerate(data):
                group.create_dataset('dataset: %s' % index, data=i, compression='gzip', compression_opts=9)
            count += sample_range + a.rp
            print count
            print("--- Sample %s: %s seconds ---" % (sample, time.time() - start_time1))

        h5f.close()

