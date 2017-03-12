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
import propagate_singlecircuit as ps
import cPickle

# Loading in Machine Learning models (should have sign data)
#####################################
y_classifier_full = joblib.load('modeldump\models_sc\sc4k_yreg_byclass.pkl')
y_class = joblib.load('modeldump\models_sc\sc4k_xaxis_class.pkl')
x_classifier_full = joblib.load('modeldump\models_sc\sc4k_xreg_byclass.pkl')
x_class = joblib.load('modeldump\models_sc\sc4k_target_xaxisrestricted.pkl')
#####################################

# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 150


# Time before the ECG starts taking measurments (should it be just 400?)
n = 2
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


def rt_ecg_gathering(ecg_list, sign_para = 'sign_para'):
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


def movingaverage(values, weight):
    sma = np.convolve(values, np.ones((weight,)) / weight, mode='same')
    return sma

def winsum(interval, window_size):
    window = np.ones(int(window_size))
    return np.convolve(interval, window, 'same')


def reg_predictor(probs,thr = 0.4):
    ws = 2
    while np.max(probs) < thr:
        probs = winsum(probs,ws)
        ws += 1
        if ws == 5:
            break
    return probs


def vecdistance(current_pos, constaints):
    """
    Translate the constraints to vector differences.
    :param current_pos: Current ECG y position
    :param constaints: The Position Constraints [Lower, Upper]
    :return: [Lower vector (+ve), Upper vector (-ve)]
    """
    lower = constaints[0]
    upper = constaints[1]
    lower_vector = None
    upper_vector = None
    if lower is None or upper is None:
        return None
    if upper >= current_pos:
        upper_vector = -(upper - current_pos)
    if upper < current_pos:
        upper_vector = -(upper + (200 % current_pos))
    if lower <= current_pos:
        lower_vector = -(lower - current_pos)
    if lower > current_pos:
        lower_vector = (current_pos + (200 % lower))
    # if current_pos == 0:
    #     lower_vector = lower
    #     upper_vector = -upper

    if lower_vector > 100:  # Largest possible vector constraints (Only happens for the x axis. Shouldn't happen for y.)
       lower_vector = 100
    if upper_vector < -99:
       upper_vector = -99

    return [lower_vector, upper_vector]


def condistance(constraint):
    """
    finds the true distance between y constraints.
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


def prediction(prob_map, vector_constraint, axis):
    """
    makes a vector prediction
    :param prob_map:
    :param vector_constraint:
    :param axis:
    :return:
    """
    if vector_constraint is None:
        possible_points = np.argwhere(prob_map == np.amax(prob_map)).flatten()
        if len(possible_points) == 1:
            return possible_points[0]
        if len(possible_points) > 1:
            return int(np.mean(possible_points))
    else:
        ref = None
        if axis == 'x':
            ref = 99
        if axis == 'y':
            ref = 176
        lower_index = ref + vector_constraint[0]
        upper_index = ref + vector_constraint[1]
        if upper_index < 0:
            upper_index = 0
        if lower_index < 0:
            lower_index = 0
        constrained_prob = prob_map[upper_index:lower_index + 1]  # create the range for examining the probabilities.
        if np.max(constrained_prob) < 0.4: #With padding this now gives an error
            return int(float(lower_index + upper_index) / 2)
        else:
            possible_points_detail = np.argwhere(constrained_prob == np.amax(constrained_prob)).flatten()
            possible_points = [x + upper_index for x in possible_points_detail]
            if len(possible_points) == 1:
                return possible_points[0]
            if len(possible_points) > 1:
                return int(np.mean(possible_points))


def constrained_finder(prev_vector, sign_short_memory_, current_ecg_pos_, constrained_, axis):
    """

    :param prev_vector:
    :param sign_short_memory_:
    :param current_ecg_pos_:
    :param constrained_:
    :param axis:
    :return:
    """
    if len(sign_short_memory_) == 1:  # Assigns the first constraint (for y case or if on boundry).
        sign = sign_short_memory_[0]

        if sign < 0:
            constrained_[0] = current_ecg_pos_

        if sign > 0:
            constrained_[1] = current_ecg_pos_

        if sign == 0:  # Starts right on the boundry
            del sign_short_memory_  # resets the short_memory as it needs to find first constraint again.
            sign_short_memory_ = []

    if len(sign_short_memory_) >= 2:  # Assigns constraints when 2 ECG have been taken.
        vsign_diff = copysign(1, sign_short_memory_[-1]) - copysign(1, sign_short_memory_[-2])
        if prev_vector < 0 and vsign_diff == 2:  # Upper Constraint
            if axis == 'x':
               if prev_vector < -4 and constrained_[0] is not None:
                   constrained_[0] += 3
                   constrained_[0] %= 200
            constrained_[1] = current_ecg_pos_

        if prev_vector > 0 and vsign_diff == -2:  # Lower Constraint
            if axis == 'x':
               if prev_vector > 4 and constrained_[1] is not None:
                   constrained_[1] -= 3
                   constrained_[1] %= 200
            constrained_[0] = current_ecg_pos_

        if prev_vector < 0 and vsign_diff == -2:  # Passed boundry (top to bottom)
            if constrained_[0] is None:
                constrained_[0] = current_ecg_pos_

        if prev_vector > 0 and vsign_diff == 2:  # Passed boundry (bottom to top)
            if constrained_[1] is None:
                constrained_[1] = current_ecg_pos_

        if prev_vector > 0 and vsign_diff == 0:  # Potential updataing of upper constraint
            if constrained_[0] is None:
                constrained_[1] = current_ecg_pos_
                if axis == 'x':
                   if prev_vector > 4:
                       constrained_[1] -= 3
                       constrained_[1] %= 200
            if condistance(constrained_) > condistance([constrained_[0], current_ecg_pos_]):
                constrained_[1] = current_ecg_pos_

        if prev_vector < 0 and vsign_diff == 0:  # Potential updating of lower constraint
            if constrained_[1] is None:
                constrained_[0] = current_ecg_pos_
                if axis == 'x':
                   if prev_vector < -4:
                       constrained_[0] += 3
                       constrained_[0] %= 200
            if condistance(constrained_) > condistance([current_ecg_pos_, constrained_[1]]):
                constrained_[0] = current_ecg_pos_

    return constrained_, sign_short_memory_

def autojump_constrainx(h_sign, constrained_, x_pos, constrainedy):
    constrained_copy = np.array(constrained_)
    if np.absolute(h_sign) < 0.8 or constrained_[1] - constrained_[0] < 50:
        return constrained_
    if constrainedy[1] != None and constrainedy[0] != None:
        if np.absolute(constrainedy[1] - constrainedy[0]) < 50:
            return constrained_
    if h_sign == 1: #Constrain right
        constrained_[1] = x_pos
    elif h_sign >= 0.8: #Constrain right with pad
        constrained_[1] = x_pos + 24
    elif h_sign == -1: #Constrain left
        constrained_[0] = x_pos
    elif h_sign <= -0.8: #Constrain left with pad
        constrained_[0] = x_pos - 24
    if constrained_[0] < constrained_copy[0]:
        constrained_[0] = constrained_copy[0]
    if constrained_[1] > constrained_copy[1]:
        constrained_[1] = constrained_copy[1]
    if constrained_[1] - np.mean(constrained_) < 24:
        m = int(np.mean(constrained_))
        constrained_[0] = m - 24
        constrained_[1] = m + 24
        if constrained_[0] < constrained_copy[0]:
            d = constrained_copy[0] - constrained_[0]
            constrained_ += d
        if constrained_[1] > constrained_copy[1]:
            d = constrained_[1] - constrained_copy[1]
            constrained_ -= d
    return constrained_


# Lists for recording data produced by algorithm
ecg_counter = [0]*number_of_rotors      # Total
ecg_start = [0]*number_of_rotors        # (x, y)
ecg_end = [0]*number_of_rotors          # (x, y)
rotor = [0]*number_of_rotors            # (x, y)
constrain_check = [0]*number_of_rotors  # num
zero_check = [0]*number_of_rotors       # num

pp = 0
print_counter(pp, number_of_rotors)
for i in range(number_of_rotors):

    # Initialising the Heart structure
    a = ps.Heart(nu=0.2, delta=0.0, fakedata=True)
    # Randomises the rotor x,y position
    cp_x_pos = randint(30, 169)
    cp_y_pos = randint(0, 199)
    a.set_circuit(np.ravel_multi_index([cp_y_pos,cp_x_pos],(200,200)))
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

    prev_y_vector = None
    prev_x_vector = None

    # Loop checking
    y_short_memory = []
    x_short_memory = []

    # vector sign history
    vsign_short_memory = []
    hsign_short_memory = []

    # Constrained y/x values
    constrainedy = [None, None]
    constrainedx = [20, 179]

    ecg_num = 0 #Num of ECGs
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
                sample = rt_ecg_gathering(process_list, sign_para='record_sign')
                 # ECG Recording and feature gathering
                ecg_num += 1

                # X AXIS FINDING
                if state == 0:
                    sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                    vsign = sample[0, :][-3]
                    hsign = sample[0, :][-2]
                    constrainedx = autojump_constrainx(h_sign = hsign, constrained_ = constrainedx, x_pos = current_ecg_x_pos, constrainedy = constrainedy)
                    con_midpoint = int(np.mean(constrainedx))
                    current_ecg_x_pos = con_midpoint
                    # sample_ = sample[0, :][0:-3].reshape(1, -1)  # Get sample without sig information.
                    sample_ = sample[0, :][0:].reshape(1, -1)
                    y_class_value = y_class.predict(sample_)[0]
                    # y_probarg = movingaverage(y_classifier_full.predict_proba(sample_)[0, :], 10)
                    y_probarg = reg_predictor(y_classifier_full.predict_proba(sample_)[0, :])

                    if y_class_value == 1:
                        state = 1  # Change to state 1 for y axis regression/classification.
                        x_class_value = x_class.predict(sample_)[0]

                        if x_class_value == 1:
                            final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                            ecg_end[i] = final_rotor_position
                            ecg_counter[i] = ecg_num
                            ECG_located_flag = True

                    if y_class_value == 0:
                        y_short_memory.append(current_ecg_y_pos)
                        vsign_short_memory.append(vsign)
                        constrainedy, vsign_short_memory = constrained_finder(prev_y_vector, vsign_short_memory,
                                                                              current_ecg_y_pos, constrainedy,axis = 'x')

                        # Tries the constrained row.
                        if condistance(constrainedy) == 1:
                            constrain_check[i] += 1
                            state = 1
                            x_class_value = x_class.predict(sample_)[0]

                            if x_class_value == 1:
                                final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                                ecg_end[i] = final_rotor_position
                                ecg_counter[i] = ecg_num
                                ECG_located_flag = True

                        else:
                            likelyp = prediction(y_probarg, vector_constraint=vecdistance(current_ecg_y_pos,
                                                 constrainedy), axis='x')
                            y_vector = y_classifier_full.classes_[likelyp]
                            prev_y_vector = y_vector
                            current_ecg_y_pos -= y_vector

                            if current_ecg_y_pos > 199 or current_ecg_y_pos < 0:
                                current_ecg_y_pos %= 200

                            # Loop Check
                            if current_ecg_y_pos in y_short_memory:
                                # final_rotor_position = ("X LOOP", "Y LOOP")
                                if y_short_memory[-1] - y_short_memory[-2] == 0:
                                   state = 1  # Change to state 1 for y axis regression/classification.
                                   zero_check[i] += 1
                                   x_class_value = x_class.predict(sample_)[0]

                                   if x_class_value == 1:
                                       final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                                       ecg_end[i] = final_rotor_position
                                       ecg_counter[i] = ecg_num
                                       ECG_located_flag = True

                                else:
                                   final_rotor_position = ("NA", "Y LOOP")
                                   ecg_end[i] = final_rotor_position
                                   ecg_counter[i] = ecg_num
                                   ECG_located_flag = True


                # Y AXIS FINDING
                if state == 1:
                    sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                    hsign = sample[0, :][-2]  # Gets the h sign
                    # sample_ = sample[0, :][0:-3].reshape(1, -1)  # Takes a sample without sign information
                    sample_ = sample[0, :][0:].reshape(1, -1)
                    x_class_value = x_class.predict(sample_)[0]
                    # x_probarg = movingaverage(x_classifier_full.predict_proba(sample_)[0, :], 10)  # Prob map
                    x_probarg = reg_predictor(x_classifier_full.predict_proba(sample_)[0, :])
                    x_class_value = x_class.predict(sample_)[0]

                    if x_class_value == 1:
                        final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                        ecg_end[i] = final_rotor_position
                        ecg_counter[i] = ecg_num
                        ECG_located_flag = True
                        constrain_check[i] = 0
                        constrainedx = [20, 179]
                        x_short_memory = []

                    if x_class_value == 0:
                        x_short_memory.append(current_ecg_x_pos)
                        hsign_short_memory.append(hsign)
                        constrainedx, hsign_short_memory = constrained_finder(prev_x_vector, hsign_short_memory,
                                                                              current_ecg_x_pos, constrainedx,axis = 'y')

                        # Tries the constrained row.
                        if condistance(constrainedx) == 1:
                            constrain_check[i] += 2
                            final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                            ecg_end[i] = final_rotor_position
                            ecg_counter[i] = ecg_num
                            ECG_located_flag = True

                        else:
                            likelyp = prediction(x_probarg, vector_constraint=vecdistance(current_ecg_x_pos,
                                                                                          constrainedx), axis='y')
                            x_vector = x_classifier_full.classes_[likelyp]
                            prev_x_vector = x_vector
                            current_ecg_x_pos -= x_vector

                            if current_ecg_x_pos > 199 or current_ecg_x_pos < 0:
                                current_ecg_x_pos %= 200

                            # Loop Check
                            if current_ecg_x_pos in x_short_memory:
                                final_rotor_position = ("X LOOP", current_ecg_y_pos)
                                if x_short_memory[-1] - x_short_memory[-2] == 0:
                                    final_rotor_position = (current_ecg_x_pos, current_ecg_y_pos)
                                    zero_check[i] += 2
                                ecg_end[i] = final_rotor_position
                                ecg_counter[i] = ecg_num
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
    print "ecg end: %s" % ecg_end
    print "ecg start: %s" % ecg_start
    print "constraint check: %s" % constrain_check
    print "zero check: %s" % zero_check


final_data = {"ECG Counter": ecg_counter, "Rotor Position": rotor, "ECG Start": ecg_start, "ECG End": ecg_end,
              "Constraint Check": constrain_check, "Zero Check": zero_check, "Machine Learning Models": ['modeldump\models_sc\sc4k_yreg_byclass.pkl', 'modeldump\models_sc\sc4k_xaxis_class.pkl', 'modeldump\models_sc\sc4k_xreg_byclass.pkl', 'modeldump\models_sc\sc4k_target_xaxisrestricted.pkl']}

if save_data == 'y':
    with open('%s.p' % save_data_name, 'wb') as f:
        cPickle.dump(final_data, f)
