"""
Runs the sign locator algorithm but without the animation (animation in pyqt_locator.py)
"""

import numpy as np
import copy
import sys
from sklearn.externals import joblib
from math import copysign
from random import randint
import analysis_theano as at
from Functions import ani_convert, feature_extract_multi_test_rt, multi_feature_compile_rt, print_counter
import propagate_singlesource as ps
import cPickle

args = sys.argv
# Loading in Machine Learning models (should have sign data)
#####################################
y_classifier_full = joblib.load(args[1])
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


def movingaverage(values, weight):
    sma = np.convolve(values, np.ones((weight,)) / weight, mode='same')
    return sma


def vecdistance(current_ypos, constaints):
    """
    Translate the constraints to vector differences.
    :param current_ypos: Current ECG y position
    :param constaints: The Position Constraints [Lower, Upper]
    :return: [Lower vector (+ve), Upper vector (-ve)]
    """
    lower = constaints[0]
    upper = constaints[1]
    lower_vector = None
    upper_vector = None
    if lower is None or upper is None:
        return None
    if upper >= current_ypos:
        upper_vector = -(upper - current_ypos)
    if upper < current_ypos:
        upper_vector = -(upper + (200 % current_ypos))
    if lower <= current_ypos:
        lower_vector = -(lower - current_ypos)
    if lower > current_ypos:
        lower_vector = (current_ypos + (200 % lower))
    if current_ypos == 0:
        lower_vector = lower
        upper_vector = -upper

    return [lower_vector, upper_vector]


def condistance(constraint):
    """

    :param constraint:
    :return:
    """
    lower = constraint[0]
    upper = constraint[1]

    if lower is None or upper is None:
        return None
    if upper > lower:
        return upper - lower
    if upper < lower:
        return upper + (200 % lower)


def prediction(prob_map, vector_constraint):
    """
    Makes a vector prediction
    :param prob_map: map given from y classifier
    :param vector_constraint: vector prediction form vecdistance
    :return:
    """
    if vector_constraint is None:
        possible_points = np.argwhere(prob_map == np.amax(prob_map)).flatten()
        print possible_points
        if len(possible_points) == 1:
            return possible_points[0]
        if len(possible_points) > 1:
            return int(np.mean(possible_points))
    else:
        lower_index = 99 + vector_constraint[0]
        upper_index = 99 + vector_constraint[1]
        constrained_prob = prob_map[upper_index:lower_index + 1]  # create the range for examining the probabilities.
        possible_points_detail = np.argwhere(constrained_prob == np.amax(constrained_prob)).flatten()
        possible_points = [x + upper_index for x in possible_points_detail]
        if len(possible_points) == 1:
            return possible_points[0]
        if len(possible_points) > 1:
            return int(np.mean(possible_points))


def rt_ecg_gathering(ecg_list, sign_para):
    """
    Records the ECGS, Gathers the features and compiles them.
    :param ecg_list: Raw data from animation_grid (t, (x,y))
    :param sign_para: decides if it records sign data
    :return: (441,) array of feature data.
    """
    voltages = ecg_processing.solve(np.array(ecg_list).astype('float32'))

    # Putting all 9 ecg features into (9,21) array
    uncompiled_features = []
    for index in range(9):
        uncompiled_features.append(feature_extract_multi_test_rt(index, voltages))
    compiled_features = multi_feature_compile_rt(np.array(uncompiled_features), sign=sign_para)
    return compiled_features


def constrainedy_finder(prev_y_vector_, vsign_short_memory_, current_ecg_y_pos_, constrainedy_):
    """

    :param prev_y_vector_:
    :param vsign_short_memory_:
    :param current_ecg_y_pos_:
    :param constrainedy_:
    :return:
    """
    if len(vsign_short_memory_) == 1:  # Assigns the first constraint.
        if vsign < 0:
            constrainedy_[0] = current_ecg_y_pos_
        if vsign > 0:
            constrainedy_[1] = current_ecg_y_pos_

    if len(vsign_short_memory_) >= 2:  # Assigns constraints when 2 ECG have been taken.
        vsign_diff = copysign(1, vsign_short_memory_[-1]) - copysign(1, vsign_short_memory_[-2])
        if prev_y_vector_ < 0 and vsign_diff == 2:  # Upper Constraint
            constrainedy_[1] = current_ecg_y_pos_

        if prev_y_vector_ > 0 and vsign_diff == -2:  # Lower Constraint
            constrainedy_[0] = current_ecg_y_pos_

        if prev_y_vector_ < 0 and vsign_diff == -2:  # Passed boundry (top to bottom)
            print "Passed boundry"
            if constrainedy_[0] is None:
                constrainedy_[0] = current_ecg_y_pos_

        if prev_y_vector_ > 0 and vsign_diff == 2:  # Passed boundry (bottom to top)
            print "Passed boundry"
            if constrainedy_[1] is None:
                constrainedy_[1] = current_ecg_y_pos_

        if prev_y_vector_ > 0 and vsign_diff == 0:  # Potential updataing of upper constraint
            # Need to work this out
            if constrainedy_[0] is None:
                constrainedy_[1] = current_ecg_y_pos_
            if condistance(constrainedy_) > condistance([constrainedy_[0], current_ecg_y_pos_]):
                constrainedy_[1] = current_ecg_y_pos_

        if prev_y_vector_ < 0 and vsign_diff == 0:  # Potential updating of lower constraint
            # Need to work this out
            if constrainedy_[1] is None:
                constrainedy_[0] = current_ecg_y_pos_
            if condistance(constrainedy_) > condistance([current_ecg_y_pos_, constrainedy_[1]]):
                constrainedy_[0] = current_ecg_y_pos_

    return constrainedy_

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
    cp_x_pos = randint(20, 180)
    cp_y_pos = randint(0, 199)
    a.set_pulse(60, [[cp_y_pos], [cp_x_pos]])
    rotor[i] = (cp_x_pos, cp_y_pos)

    # Initialising ECG recording (randomises the probe x,y position)
    current_ecg_x_pos = randint(20, 180)
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

    prev_y_vector = None

    # vector sign history
    vsign_short_memory = []

    # Constrained y values
    constrainedy = [None, None]

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
                    sample = rt_ecg_gathering(process_list, sign_para=args[5])  # ECG Recording and feature gathering
                    y_ecg_num += 1
                    sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                    y_class_value = y_estimator.predict(sample)[0]
                    y_probarg = y_classifier_full.predict_proba(sample)[0, :]
                    vsign = sample[0, :][-3]

                    if y_class_value == 1:
                        state = 1  # Change to state 1 for y axis regression/classification.
                        del y_short_memory  # Checks for loops
                        y_short_memory = []
                        del vsign_short_memory
                        vsign_short_memory = []
                        del constrainedy
                        constrainedy = [None, None]
                        x_class_value = x_class.predict(sample)[0]

                        if x_class_value == 1:
                            final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                            ecg_end[i] = final_rotor_position
                            ecg_counter[i] = (y_ecg_num, 0)
                            ECG_located_flag = True

                    if y_class_value == 0:

                        y_short_memory.append(current_ecg_y_pos)
                        vsign_short_memory.append(vsign)

                        # if len(vsign_short_memory) == 1:  # Assigns the first constraint.
                        #     if vsign < 0:
                        #         constrainedy[0] = current_ecg_y_pos
                        #     if vsign > 0:
                        #         constrainedy[1] = current_ecg_y_pos
                        #
                        # if len(vsign_short_memory) >= 2:  # Assigns constraints when 2 ECG have been taken.
                        #     vsign_diff = copysign(1, vsign_short_memory[-1]) - copysign(1, vsign_short_memory[-2])
                        #     if prev_y_vector < 0 and vsign_diff == 2:  # Upper Constraint
                        #         constrainedy[1] = current_ecg_y_pos
                        #
                        #     if prev_y_vector > 0 and vsign_diff == -2:  # Lower Constraint
                        #         constrainedy[0] = current_ecg_y_pos
                        #
                        #     if prev_y_vector < 0 and vsign_diff == -2:  # Passed boundry (top to bottom)
                        #         print "Passed boundry"
                        #         if constrainedy[0] is None:
                        #             constrainedy[0] = current_ecg_y_pos
                        #
                        #     if prev_y_vector > 0 and vsign_diff == 2:  # Passed boundry (bottom to top)
                        #         print "Passed boundry"
                        #         if constrainedy[1] is None:
                        #             constrainedy[1] = current_ecg_y_pos
                        #
                        #     if prev_y_vector > 0 and vsign_diff == 0:  # Potential updataing of upper constraint
                        #         # Need to work this out
                        #         if constrainedy[0] is None:
                        #             constrainedy[1] = current_ecg_y_pos
                        #         if condistance(constrainedy) > condistance([constrainedy[0], current_ecg_y_pos]):
                        #             constrainedy[1] = current_ecg_y_pos
                        #
                        #     if prev_y_vector < 0 and vsign_diff == 0:  # Potential updating of lower constraint
                        #         # Need to work this out
                        #         if constrainedy[1] is None:
                        #             constrainedy[0] = current_ecg_y_pos
                        #         if condistance(constrainedy) > condistance([current_ecg_y_pos, constrainedy[1]]):
                        #             constrainedy[0] = current_ecg_y_pos

                        constrainedy = constrainedy_finder(prev_y_vector, vsign_short_memory,
                                                           current_ecg_y_pos, constrainedy)

                        print "constrainedy: %s" % constrainedy
                        likelyp = prediction(y_probarg, vector_constraint=vecdistance(current_ecg_y_pos, constrainedy))
                        y_vector = y_classifier_full.classes_[likelyp]
                        prev_y_vector = y_vector
                        print "y vector: %s" % y_vector
                        current_ecg_y_pos -= y_vector

                        if current_ecg_y_pos > 200 or current_ecg_y_pos < 0:
                            current_ecg_y_pos %= 200

                        # Loop Check (might not need this anymore)
                        # if current_ecg_y_pos in y_short_memory:
                        #     final_rotor_position = "Y LOOP"
                        #     ecg_end[i] = final_rotor_position
                        #     ecg_counter[i] = (y_ecg_num, 0)
                        #     ECG_located_flag = True

                # if state == 1:
                #     sample = rt_ecg_gathering(process_list, sign_para='no sign')  # ECG Recording and feature gathering
                #     x_ecg_num += 1
                #     sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                #
                #     x_class_value = x_class.predict(sample)[0]
                #     x_vector = int(x_regress.predict(sample)[0])
                #
                #     if x_class_value == 1:
                #         final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                #         ecg_end[i] = final_rotor_position
                #         ecg_counter[i] = (y_ecg_num, x_ecg_num)
                #         ECG_located_flag = True
                #         del x_short_memory  # Checks for loops
                #         x_short_memory = []
                #
                #     if x_class_value == 0:
                #         x_short_memory.append(current_ecg_x_pos)
                #         current_ecg_x_pos -= x_vector
                #         if current_ecg_x_pos > 200 or current_ecg_x_pos < 0:
                #             current_ecg_x_pos %= 200
                #
                #         # Loop Check
                #         if current_ecg_x_pos in x_short_memory:
                #             final_rotor_position = "X LOOP"
                #             ecg_end[i] = final_rotor_position
                #             ecg_counter[i] = (y_ecg_num, x_ecg_num)
                #             ECG_located_flag = True

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
