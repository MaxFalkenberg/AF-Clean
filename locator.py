"""
Runs the simple locator algorithm but without the animation (animation in pyqt_locator.py)
"""

import numpy as np
import copy
import sys
from sklearn.externals import joblib
from random import randint
import analysis_theano as at
from Functions import ani_convert, feature_extract_multi_test_rt, multi_feature_compile_rt, print_counter
import propagate_singlesource as ps
import cPickle

args = sys.argv
# Loading in Machine Learning models
#####################################
y_regress = joblib.load(args[1])
y_estimator = joblib.load(args[2])
x_regress = joblib.load(args[3])
x_class = joblib.load(args[4])
#####################################

# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 360

# Time before the ECG starts taking measurments (should it be just 400?)
n = 1
stability_time = n * process_length


# Number of rotors to find
number_of_rotors = int(raw_input('Number of rotors to find: '))

while True:
    save_data = raw_input("Save rotor location data (y/n): ")
    if save_data in ['y', 'n']:
        break

save_data_name = None
if save_data == 'y':
    save_data_name = raw_input("Saved datafile name: ")


def rt_ecg_gathering(ecg_list):
    """
    Records the ECGS, Gathers the features and compiles them.
    :param ecg_list: Raw data from animation_grid (t, (x,y))
    :return: (441,) array of feature data.
    """
    voltages = ecg_processing.solve(np.array(ecg_list).astype('float32'))

    # Putting all 9 ecg features into (9,21) array
    uncompiled_features = []
    for index in range(9):
        uncompiled_features.append(feature_extract_multi_test_rt(index, voltages))
    compiled_features = multi_feature_compile_rt(np.array(uncompiled_features), sign=args[5])
    return compiled_features


# Lists for recording data produced by algorithm
ecg_counter = [0]*number_of_rotors
ecg_start = [0]*number_of_rotors
ecg_end = [0]*number_of_rotors
rotor = [0]*number_of_rotors

pp = 0
print_counter(pp, number_of_rotors)
for i in range(number_of_rotors):

    # Initialising the Heart structure
    a = ps.Heart(nu=0.2, delta=0.0, fakedata=True)
    # Randomises the rotor x,y position
    cp_x_pos = randint(30, 169)
    cp_y_pos = randint(0, 199)
    a.set_pulse(60, [[cp_y_pos], [cp_x_pos]])
    rotor[i] = (cp_x_pos, cp_y_pos)

    # Initialising ECG recording (randomises the probe x,y position)
    current_ecg_x_pos = randint(20, 179)
    current_ecg_y_pos = randint(0, 199)
    ecg_processing = at.ECG(centre=(current_ecg_y_pos, current_ecg_x_pos), m='g_single')
    ecg_start[i] = (current_ecg_x_pos, current_ecg_y_pos)

    # Animation grids
    animation_grid = np.zeros(a.shape)

    # reset time step counter
    ptr1 = 0

    # Loop checking
    y_short_memory = []
    x_short_memory = []

    y_ecg_num = 0  # Number of y ecgs
    x_ecg_num = 0  # Number of x ecgs
    process_list = []  # process list
    final_rotor_position = None  # Final rotor position tuple
    ECG_start_flag = False  # Setting measurment flag to False (ecg measurments start when flag is triggered).
    ECG_located_flag = False  # To keep the system propogating until the rotor is found or limit is reached.

    state = 0  # State for pipe work.

    while not ECG_located_flag:

        # Propogates model and converts into ani_data (used in ecg data processing)
        data = a.propagate(ecg=True)
        data = ani_convert(data, shape=a.shape, rp=a.rp, animation_grid=animation_grid)

        # If flag triggered, then start taking measurments.
        if ECG_start_flag:
            process_list.append(copy.deepcopy(data))

        ptr1 += 1

        if ptr1 >= stability_time:
            if not ECG_start_flag:
                ECG_start_flag = True

            if ptr1 % process_length == 0 and ptr1 != stability_time:

                if state == 0:
                    sample = rt_ecg_gathering(process_list)  # ECG Recording and feature gathering
                    y_ecg_num += 1
                    sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                    y_class_value = y_estimator.predict(sample)[0]
                    y_vector = int(y_regress.predict(sample)[0])

                    if y_class_value == 1:
                        state = 1  # Change to state 1 for y axis regression/classification.
                        del y_short_memory  # Checks for loops
                        y_short_memory = []
                        x_class_value = x_class.predict(sample)[0]

                        if x_class_value == 1:
                            final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                            ecg_end[i] = final_rotor_position
                            ecg_counter[i] = (y_ecg_num, 0)
                            ECG_located_flag = True

                    if y_class_value == 0:
                        y_short_memory.append(current_ecg_y_pos)
                        current_ecg_y_pos -= y_vector
                        if current_ecg_y_pos > 200 or current_ecg_y_pos < 0:
                            current_ecg_y_pos %= 200

                        # Loop Check
                        if current_ecg_y_pos in y_short_memory:
                            final_rotor_position = "Y LOOP"
                            ecg_end[i] = final_rotor_position
                            ecg_counter[i] = (y_ecg_num, 0)
                            ECG_located_flag = True

                if state == 1:
                    sample = rt_ecg_gathering(process_list)  # ECG Recording and feature gathering
                    x_ecg_num += 1
                    sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.

                    x_class_value = x_class.predict(sample)[0]
                    x_vector = int(x_regress.predict(sample)[0])

                    if x_class_value == 1:
                        final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                        ecg_end[i] = final_rotor_position
                        ecg_counter[i] = (y_ecg_num, x_ecg_num)
                        ECG_located_flag = True
                        del x_short_memory  # Checks for loops
                        x_short_memory = []

                    if x_class_value == 0:
                        x_short_memory.append(current_ecg_x_pos)
                        current_ecg_x_pos -= x_vector
                        if current_ecg_x_pos > 200 or current_ecg_x_pos < 0:
                            current_ecg_x_pos %= 200

                        # Loop Check
                        if current_ecg_x_pos in x_short_memory:
                            final_rotor_position = "X LOOP"
                            ecg_end[i] = final_rotor_position
                            ecg_counter[i] = (y_ecg_num, x_ecg_num)
                            ECG_located_flag = True

                del process_list
                process_list = []
                ecg_processing.reset_singlegrid((current_ecg_y_pos, current_ecg_x_pos))

    pp += 1
    print_counter(pp, number_of_rotors)

print '\n'
if save_data == 'n':
    print "ecg counter: %s" % ecg_counter
    print "rotor position: %s" % rotor
    print "ecg start: %s" % ecg_start
    print "ecg end: %s" % ecg_end

final_data = {"ECG Counter": ecg_counter, "Rotor Position": rotor, "ECG Start": ecg_start, "ECG End": ecg_end,
              "Machine Learning Models": [args[1], args[2], args[3], args[4]]}

if save_data == 'y':
    with open('%s.p' % save_data_name, 'wb') as f:
        cPickle.dump(final_data, f)
