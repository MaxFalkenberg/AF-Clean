"""
Runs the sign locator algorithm but without the animation (animation in pyqt_locator.py)
"""

import numpy as np
import copy
import sys
from sklearn.externals import joblib
from math import copysign
from random import randint, choice
import analysis_theano as at
from Functions import ani_convert, feature_extract_multi_test_rt, multi_feature_compile_rt,\
    print_counter, check_bsign, check_signs
import propagate_singlecircuit as ps
import cPickle

args = sys.argv
# Loading in Machine Learning models (should have sign data)
#####################################
y_classifier_full = joblib.load(args[1])
y_class = joblib.load(args[2])
x_classifier_full = joblib.load(args[3])
x_class = joblib.load(args[4])
vsign_check = np.load('/Users/williamcheng/AF-Clean/vsign_tensor.npy')
hsign_check = np.load('/Users/williamcheng/AF-Clean/hsign_tensor.npy')
axessign_check = np.load('/Users/williamcheng/AF-Clean/axessign_tensor.npy')

# y_classifier_full = joblib.load('modeldump\models_sc\sc4k_yreg_byclass.pkl')
# y_class = joblib.load('modeldump\models_sc\sc4k_xaxis_class.pkl')
# x_classifier_full = joblib.load('modeldump\models_sc\sc4k_xreg_byclass.pkl')
# x_class = joblib.load('modeldump\models_sc\sc4k_target_xaxisrestricted.pkl')
# vsign_check = np.load('vsign_tensor.npy')
# hsign_check = np.load('hsign_tensor.npy')
# axessign_check = np.load('axessign_tensor.npy')
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
    compiled_features, signs = multi_feature_compile_rt(np.array(uncompiled_features), sign='record_sign_plus')
    return compiled_features, signs


def movingaverage(values, weight):
    sma = np.convolve(values, np.ones((weight,)) / weight, mode='same')
    return sma


def winsum(interval, window_size):
    window = np.ones(int(window_size))
    return np.convolve(interval, window, 'same')


def reg_predictor(probs, thr=0.4):
    ws = 2
    while np.max(probs) < thr:
        probs = winsum(probs, ws)
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
    if upper == lower:
        return 0


def conposition(lower, upper):
    """
    Determines the new possible range of possible y positions.
    :param upper:
    :param lower:
    :return:
    """
    if upper > lower:
        return range(lower, upper)
    if lower > upper:
        return range(lower, 200) + range(0, upper)


def relative_vectors(x_pos, y_pos, ref_x, ref_y):
    """
    give back the relative vector in relation to the reference point.
    :param x_pos:
    :param y_pos:
    :param ref_x:
    :param ref_y:
    :return:
    """
    vector_x = [(x - ref_x) for x in x_pos]
    vector_y = [(y - ref_y) for y in y_pos]
    vector_y = [y-200 if y > 100 else y+200 if y <= -100 else y for y in vector_y]
    return vector_x, vector_y


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
        constrained_prob = prob_map[upper_index:lower_index + 1]  # create the range for examining the probabilities.
        if np.max(constrained_prob) < 0.4:
            return int(float(lower_index + upper_index) / 2)
        else:
            possible_points_detail = np.argwhere(constrained_prob == np.amax(constrained_prob)).flatten()
            possible_points = [x + upper_index for x in possible_points_detail]
            if len(possible_points) == 1:
                return int(possible_points[0])
            if len(possible_points) > 1:
                return int(np.mean(possible_points))


def constrained_finder(prev_vector, sign_short_memory_, current_ecg_pos_, constrained_, axis, perm_constraints):
    """

    :param prev_vector:
    :param sign_short_memory_:
    :param current_ecg_pos_:
    :param constrained_:
    :param axis:
    :param perm_constraints:
    :param special:
    :return:
    """
    if len(sign_short_memory_) == 1:  # Assigns the first constraint (for y case or if on boundry).
        sign = sign_short_memory_[0]
        if sign < 0:
            constrained_[0] = current_ecg_pos_
            if perm_constraints and axis == 'x':
                constrained_[1] = perm_constraints[0][1]

        if sign > 0:
            constrained_[1] = current_ecg_pos_
            if perm_constraints and axis == 'x':
                constrained_[0] = perm_constraints[0][0]

        if sign == 0:  # Starts right on the boundry
            if not perm_constraints:
                del sign_short_memory_  # resets the short_memory as it needs to find first constraint again.
                sign_short_memory_ = []
            else:
                constrained_[0] = perm_constraints[0][0]
                constrained_[1] = perm_constraints[0][1]

    if len(sign_short_memory_) >= 2:  # Assigns constraints when 2 ECG have been taken.
        vsign_diff = int(copysign(1, sign_short_memory_[-1]) - copysign(1, sign_short_memory_[-2]))

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


def special_constraint_finder(current_x, current_y, total_sign, constrained_y, contrained_x, perm_const,spec_y,spec_v):
    """

    :param current_y:
    :param current_x:
    :param total_sign:
    :param constrained_y:
    :param contrained_x:
    :param perm_const:
    :param pred_y:
    :return:
    """
    b = check_bsign(total_sign[0][-2], total_sign[0][-1])
    h = total_sign_info[0][1]
    v = total_sign_info[0][0]
    if b != 0:
        pass
    else:
        if v >= 0:
            constrained_y[1] = current_y
            constrained_y[0] = perm_const[0][0]
            p = True

        if v < 0:
            constrained_y[1] = perm_const[0][1]
            constrained_y[0] = current_y
            p = False

        if np.size(spec_y) != 0:
            u = constrained_y[1]
            l = constrained_y[0]
            for i in range(len(spec_y)):
                m = spec_y[i]
                if (u-m) * (m-l) >= 0:
                    if p:
                        constrained_y[0] = m
                    else:
                        constrained_y[1] = m
                    break

        # if h == -1:
        #     contrained_x[0] = current_x
        #
        # if h == 1:
        #     contrained_x[1] = current_x

    x_jump = True

    return contrained_x, constrained_y, x_jump


def constrained_finder(prev_vector, sign_short_memory_, current_ecg_pos_, constrained_, axis, perm_constraints):
    """

    :param prev_vector:
    :param sign_short_memory_:
    :param current_ecg_pos_:
    :param constrained_:
    :param axis:
    :param perm_constraints:
    :return:
    """
    if len(sign_short_memory_) == 1:  # Assigns the first constraint (for y case or if on boundry).
        sign = sign_short_memory_[0]
        if sign < 0:
            constrained_[0] = current_ecg_pos_
            if perm_constraints and axis == 'x':
                constrained_[1] = perm_constraints[0][1]

        if sign > 0:
            constrained_[1] = current_ecg_pos_
            if perm_constraints and axis == 'x':
                constrained_[0] = perm_constraints[0][0]

        if sign == 0:  # Starts right on the boundry
            if not perm_constraints:
                del sign_short_memory_  # resets the short_memory as it needs to find first constraint again.
                sign_short_memory_ = []
            else:
                constrained_[0] = perm_constraints[0][0]
                constrained_[1] = perm_constraints[0][1]

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
    """

    :param prev_vector:
    :param sign_short_memory_:
    :param current_ecg_pos_:
    :param constrained_:
    :param axis_target:
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
            if axis_target == 'x':
                if prev_vector < -3 and constrained_[0] is not None:
                    constrained_[0] += 3
                    constrained_[0] %= 200
            constrained_[1] = current_ecg_pos_

        if prev_vector > 0 and vsign_diff == -2:  # Lower Constraint
            if axis_target == 'x':
                if prev_vector > 3 and constrained_[1] is not None:
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
                if axis_target == 'x':
                    if prev_vector > 3:
                        constrained_[1] -= 3
                        constrained_[1] %= 200
            if condistance(constrained_) > condistance([constrained_[0], current_ecg_pos_]):
                constrained_[1] = current_ecg_pos_

        if prev_vector < 0 and vsign_diff == 0:  # Potential updating of lower constraint
            if constrained_[1] is None:
                constrained_[0] = current_ecg_pos_
                if axis_target == 'x':
                    if prev_vector < -3:
                        constrained_[0] += 3
                        constrained_[0] %= 200
            if condistance(constrained_) > condistance([current_ecg_pos_, constrained_[1]]):
                constrained_[0] = current_ecg_pos_

    return constrained_, sign_short_memory_

# Lists for recording data produced by algorithm
ecg_counter = [0]*number_of_rotors      # Total
ecg_start = [0]*number_of_rotors        # (x, y)
ecg_end = [0]*number_of_rotors          # (x, y)
rotor = [0]*number_of_rotors            # (x, y)
yloop = [0]*number_of_rotors
xloop = [0]*number_of_rotors
constrain_check_y = [0]*number_of_rotors  # num
constrain_check_x = [0]*number_of_rotors  # num
zero_y_check = [0]*number_of_rotors       # num
zero_x_check = [0]*number_of_rotors       # num

pp = 0
print_counter(pp, number_of_rotors)
for tissue in range(number_of_rotors):

    # Initialising the Heart structure
    a = ps.Heart(nu=0.2, delta=0.0, fakedata=True)
    # Randomises the rotor x,y position
    cp_x_pos = randint(30, 169)
    cp_y_pos = randint(0, 199)
    cp_x_pos2 = randint(30, 169)
    cp_y_pos2 = randint(0, 199)
    while np.absolute(cp_y_pos2 - cp_y_pos) <= 10 or 200 - np.absolute(cp_y_pos2 - cp_y_pos) <= 10:
        cp_x_pos = randint(30, 169)
        cp_y_pos = randint(0, 199)
        cp_x_pos2 = randint(30, 169)
        cp_y_pos2 = randint(0, 199)
    a.set_multi_circuit(np.ravel_multi_index([cp_y_pos, cp_x_pos], (200, 200)),
                        np.ravel_multi_index([cp_y_pos2, cp_x_pos2], (200, 200)))

    rotor[tissue] = [(cp_x_pos, cp_y_pos), (cp_x_pos2, cp_y_pos2)]

    pred_rotor_x = []
    pred_rotor_y = []

    # Rotor state
    rotors_found = 0

    # estimated number of rotors (Needs a function to predict this value)
    N_rotors = 1

    # Preparing the recorded ECGS
    ecg_end_positions = [0] * (N_rotors + 1)

    # Initialising ECG recording (randomises the probe x,y position)
    current_ecg_x_pos = randint(20, 179)
    current_ecg_y_pos = randint(0, 199)
    ecg_processing = at.ECG(centre=(current_ecg_y_pos, current_ecg_x_pos), m='g_single')
    ecg_start[tissue] = (current_ecg_x_pos, current_ecg_y_pos)

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
    vconsistent = []
    hconsistent = []
    bconsistent = []

    special_state = False
    jump_x = False

    # History information
    total_sign_info = []
    x_history = []
    y_history = []

    # Constrained y/x values
    constrainedy = [None, None]
    constrainedx = [20, 179]

    current_constraint_y_info = [0] * (N_rotors + 1)
    current_constraint_x_info = [0] * (N_rotors + 1)

    zero_y_info = [0] * (N_rotors + 1)
    zero_x_info = [0] * (N_rotors + 1)

    yloop_count = [0] * (N_rotors + 1)
    xloop_count = [0] * (N_rotors + 1)

    # List of perminant constraints (gets reset when all rotors are found)
    perminant_constraints = []

    ecg_num_list = [0] * (N_rotors + 1)
    ecg_num = 0  # Number of ecgs
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
                sample, signs = rt_ecg_gathering(process_list, sign_para='record_sign_plus')
                # ECG Recording and feature gathering
                ecg_num += 1
                total_sign_info.append(signs)
                x_history.append(current_ecg_x_pos)
                y_history.append(current_ecg_y_pos)

                # LOOKING FOR THE ROTORS X AXIS
                if state == 0:
                    sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                    vsign = sample[0, :][-3]
                    y_class_value = y_class.predict(sample)[0]
                    y_probarg = reg_predictor(y_classifier_full.predict_proba(sample)[0, :])

                    # POSITIVE CLASSIFICATION FOR Y
                    if y_class_value == 1:
                        if special_state:
                            special_state = False
                        state = 1  # Change to state 1 for y axis regression/classification.
                        x_class_value = x_class.predict(sample)[0]

                        # CHECKS IF ON THE CORRECT Y AXIS
                        if x_class_value == 1:
                            ecg_end_positions[rotors_found] = (current_ecg_x_pos, current_ecg_y_pos)
                            ecg_num_list[rotors_found] = ecg_num
                            state = 0
                            pred_rotor_y.append(current_ecg_y_pos)
                            pred_rotor_x.append(current_ecg_x_pos)

                            # ALL ROTORS ARE FOUND
                            if rotors_found == N_rotors:
                                ecg_end_positions[rotors_found] = (current_ecg_x_pos, current_ecg_y_pos)
                                rotors_found = 0
                                ecg_end[tissue] = ecg_end_positions
                                ecg_counter[tissue] = ecg_num_list
                                constrain_check_y[tissue] = current_constraint_y_info
                                constrain_check_x[tissue] = current_constraint_x_info
                                xloop[tissue] = xloop_count
                                yloop[tissue] = yloop_count
                                zero_x_check[tissue] = zero_x_info
                                zero_y_check[tissue] = zero_y_info
                                ECG_located_flag = True

                            # ONE OF THE ROTORS IS FOUND
                            else:
                                rotors_found += 1
                                upper = current_ecg_y_pos - 5
                                upper %= 200
                                lower = current_ecg_y_pos + 5
                                lower %= 200
                                perminant_constraints.append([lower, upper])
                                xvec, yvec = relative_vectors(x_history, y_history, current_ecg_x_pos,
                                                              current_ecg_y_pos)
                                xvec = np.array(xvec)
                                yvec = np.array(yvec)

                                for i in range(len(x_history)):
                                    vconsistent.append(
                                        check_signs(xvec[i], yvec[i], total_sign_info[i][0], vsign_check, thr=0.05))
                                    hconsistent.append(
                                        check_signs(xvec[i], yvec[i], total_sign_info[i][1], hsign_check, thr=0.02))
                                    bconsistent.append(check_bsign(total_sign_info[i][-2], total_sign_info[i][-1]))
                                vconsistent = np.array(vconsistent)[np.absolute(yvec) > 4]
                                hconsistent = np.array(hconsistent)[np.absolute(yvec) > 4]
                                bconsistent = np.array(bconsistent)[np.absolute(yvec) > 4]
                                total_sign_info = np.array(total_sign_info)
                                tsign = total_sign_info[np.absolute(yvec) > 4]
                                yvec_use = yvec[np.absolute(yvec) > 4]
                                xvec_use = xvec[np.absolute(yvec) > 4]
                                special_y = (yvec_use[vconsistent == False] + current_ecg_y_pos) % 200
                                special_vsign = tsign[:, 0][vconsistent == False]

                                current_ecg_y_pos = (current_ecg_y_pos + 100) % 200
                                special_state = True
                                y_short_memory = []
                                vsign_short_memory = []
                                total_sign_info = []
                                vconsistent = []
                                hconsistent = []
                                bconsistent = []
                                x_history = []
                                y_history = []
                                ecg_num = 0

                            constrainedy = [None, None]
                            constrainedx = [20, 179]

                    # NEGATIVE CLASSIFICATION FOR Y
                    if y_class_value == 0:
                        y_short_memory.append(current_ecg_y_pos)
                        vsign_short_memory.append(vsign)
                        if not special_state:
                            constrainedy, vsign_short_memory = constrained_finder(prev_y_vector, vsign_short_memory,
                                                                                  current_ecg_y_pos, constrainedy, 'x',
                                                                                  perminant_constraints)
                        if special_state:
                            constrainedx, constrainedy, jump_x = special_constraint_finder(current_ecg_x_pos,
                                                                                           current_ecg_y_pos,
                                                                                           total_sign_info,
                                                                                           constrainedy,
                                                                                           constrainedx,
                                                                                           perminant_constraints,
                                                                                           special_y, special_vsign)
                            # might need this to fix weird constraints
                            vsign_short_memory = []

                        # CONSTRAINED CONDITION FOR Y
                        if condistance(constrainedy) == 0:
                            current_constraint_y_info[rotors_found] += 1
                            if rotors_found > 0:
                                current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                            else:
                                current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            y_short_memory = []
                            vsign_short_memory = []
                            total_sign_info = []
                            vconsistent = []
                            hconsistent = []
                            bconsistent = []
                            x_history = []
                            y_history = []

                        # MOVING THE PROBE IN THE Y AXIS
                        else:
                            likelyp = prediction(y_probarg, vector_constraint=vecdistance(current_ecg_y_pos,
                                                                                          constrainedy), axis='x')
                            y_vector = y_classifier_full.classes_[likelyp]

                            # SPECIAL CONDITION
                            if np.abs(y_vector) > 45 and special_state:
                                y_vector = int(45 * copysign(1, y_vector))
                            special_state = False

                            # JUMPS IN THE X DIRECTION AS WELL UNDER SPECIAL CONDITION
                            if jump_x:
                                h = total_sign_info[-1][1]
                                if np.abs(h) > 1.0:
                                    if h > 0:
                                        current_ecg_x_pos = int((current_ecg_x_pos + 20) / 2.)
                                    else:
                                        current_ecg_x_pos = int((current_ecg_x_pos + 179) / 2.)

                                else:
                                    if 179 - current_ecg_x_pos >= 80:
                                        current_ecg_x_pos += 80
                                    else:
                                        current_ecg_x_pos -= 80
                                jump_x = False

                            if rotors_found > 0 and condistance(constrainedy) < 60:
                                d = condistance(constrainedy)
                                if type(d) == type(None):
                                    d = 100
                                if np.abs(y_vector) > d * 0.75:
                                    y_vector = int(np.sign(y_vector) * d / 2.)

                            # IF THE PREDICTED Y JUMP IS ZERO
                            if y_vector == 0:
                                state = 0
                                zero_y_info[rotors_found] += 1
                                constrainedy = [None, None]
                                constrainedx = [20, 179]
                                if rotors_found > 0:
                                    current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                                else:
                                    current_ecg_y_pos = randint(0, 199)
                                current_ecg_x_pos = randint(20, 179)
                                y_short_memory = []
                                vsign_short_memory = []

                            prev_y_vector = y_vector
                            current_ecg_y_pos -= y_vector

                            # WRAPPING AROUND THE BOUNDRY CONDITION
                            if current_ecg_y_pos > 199 or current_ecg_y_pos < 0:
                                current_ecg_y_pos %= 200

                            # Y LOOP FORM CHECK
                            if current_ecg_y_pos in y_short_memory:
                                yloop_count[rotors_found] += 1
                                constrainedy = [None, None]
                                constrainedx = [20, 179]
                                if rotors_found > 0:
                                    current_ecg_y_pos = (current_ecg_y_pos + 100) % 200
                                else:
                                    current_ecg_y_pos = randint(0, 199)
                                current_ecg_x_pos = randint(20, 179)
                                y_short_memory = []
                                vsign_short_memory = []
                                total_sign_info = []
                                vconsistent = []
                                hconsistent = []
                                bconsistent = []
                                x_history = []
                                y_history = []

                # LOOKING FOR THE ROTORS Y AXIS
                if state == 1:
                    sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                    hsign = sample[0, :][-2]  # Gets the h sign
                    x_probarg = reg_predictor(x_classifier_full.predict_proba(sample)[0, :])  # Prob map
                    x_class_value = x_class.predict(sample)[0]

                    # POSITIVE CLASSIFICATION FOR X
                    if x_class_value == 1:
                        ecg_end_positions[rotors_found] = (current_ecg_x_pos, current_ecg_y_pos)
                        ecg_num_list[rotors_found] = ecg_num
                        state = 0
                        pred_rotor_y.append(current_ecg_y_pos)
                        pred_rotor_x.append(current_ecg_x_pos)

                        # ALL ROTORS ARE FOUND
                        if rotors_found == N_rotors:
                            ecg_end_positions[rotors_found] = (current_ecg_x_pos, current_ecg_y_pos)
                            rotors_found = 0
                            ecg_end[tissue] = ecg_end_positions
                            ecg_counter[tissue] = ecg_num_list
                            constrain_check_y[tissue] = current_constraint_y_info
                            constrain_check_x[tissue] = current_constraint_x_info
                            xloop[tissue] = xloop_count
                            yloop[tissue] = yloop_count
                            zero_x_check[tissue] = zero_x_info
                            zero_y_check[tissue] = zero_y_info
                            ECG_located_flag = True

                        # ONE OF THE ROTORS IS FOUND
                        else:
                            rotors_found += 1
                            upper = current_ecg_y_pos - 5
                            upper %= 200
                            lower = current_ecg_y_pos + 5
                            lower %= 200
                            perminant_constraints.append([lower, upper])
                            xvec, yvec = relative_vectors(x_history, y_history, current_ecg_x_pos,
                                                          current_ecg_y_pos)
                            xvec = np.array(xvec)
                            yvec = np.array(yvec)

                            for i in range(len(x_history)):
                                vconsistent.append(
                                    check_signs(xvec[i], yvec[i], total_sign_info[i][0], vsign_check, thr=0.05))
                                hconsistent.append(
                                    check_signs(xvec[i], yvec[i], total_sign_info[i][1], hsign_check, thr=0.02))
                                bconsistent.append(check_bsign(total_sign_info[i][-2], total_sign_info[i][-1]))
                            vconsistent = np.array(vconsistent)[np.absolute(yvec) > 4]
                            hconsistent = np.array(hconsistent)[np.absolute(yvec) > 4]
                            bconsistent = np.array(bconsistent)[np.absolute(yvec) > 4]
                            total_sign_info = np.array(total_sign_info)
                            tsign = total_sign_info[np.absolute(yvec) > 4]
                            yvec_use = yvec[np.absolute(yvec) > 4]
                            xvec_use = xvec[np.absolute(yvec) > 4]
                            special_y = (yvec_use[vconsistent == False] + current_ecg_y_pos) % 200
                            special_vsign = tsign[:, 0][vconsistent == False]

                            current_ecg_y_pos = (current_ecg_y_pos + 100) % 200
                            special_state = True
                            y_short_memory = []
                            vsign_short_memory = []
                            total_sign_info = []
                            x_history = []
                            y_history = []
                            ecg_num = 0

                        constrainedy = [None, None]
                        constrainedx = [20, 179]

                    # NEGATIVE CLASSIFICATION FOR X
                    if x_class_value == 0:
                        x_short_memory.append(current_ecg_x_pos)
                        hsign_short_memory.append(hsign)
                        constrainedx, hsign_short_memory = constrained_finder(prev_x_vector, hsign_short_memory,
                                                                              current_ecg_x_pos, constrainedx,
                                                                              'y', perminant_constraints)

                        # CONSTRAINED CONDITION FOR X
                        if condistance(constrainedx) == 0:
                            state = 0
                            current_constraint_x_info[rotors_found] += 1
                            constrainedy = [None, None]
                            constrainedx = [20, 179]
                            if rotors_found > 0:
                                current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                            else:
                                current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            x_short_memory = []
                            hsign_short_memory = []
                            total_sign_info = []
                            x_history = []
                            y_history = []
                            vconsistent = []
                            hconsistent = []
                            bconsistent = []

                        # MOVING THE PROBE IN THE X AXIS
                        else:
                            likelyp = prediction(x_probarg,
                                                 vector_constraint=vecdistance(current_ecg_x_pos, constrainedx),
                                                 axis='y')
                            x_vector = x_classifier_full.classes_[likelyp]

                            # IF THE PREDICTED JUMP IS ZERO
                            if x_vector == 0:
                                state = 0
                                zero_x_info[rotors_found] += 1
                                constrainedy = [None, None]
                                constrainedx = [20, 179]
                                if rotors_found > 0:
                                    current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                                else:
                                    current_ecg_y_pos = randint(0, 199)
                                current_ecg_x_pos = randint(20, 179)
                                x_short_memory = []
                                hsign_short_memory = []
                                total_sign_info = []
                                x_history = []
                                y_history = []
                                vconsistent = []
                                hconsistent = []
                                bconsistent = []

                            prev_x_vector = x_vector
                            current_ecg_x_pos -= x_vector

                            # WRAPPING AROUND THE BOUNDRY
                            if current_ecg_x_pos > 199 or current_ecg_x_pos < 0:
                                current_ecg_x_pos %= 200

                            # X LOOP FORM CHECK
                            if current_ecg_x_pos in x_short_memory:
                                state = 0
                                xloop_count[rotors_found] += 1
                                constrainedy = [None, None]
                                constrainedx = [20, 179]
                                if rotors_found > 0:
                                    current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                                else:
                                    current_ecg_y_pos = randint(0, 199)
                                current_ecg_x_pos = randint(20, 179)
                                x_short_memory = []
                                hsign_short_memory = []
                                total_sign_info = []
                                x_history = []
                                y_history = []
                                vconsistent = []
                                hconsistent = []
                                bconsistent = []

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
    print "Y constraint check: %s" % constrain_check_y
    print "X constraint check: %s" % constrain_check_x
    print "Y zero check: %s" % zero_y_check
    print "X zero check: %s" % zero_x_check

final_data = {"ECG Counter": ecg_counter, "Rotor Position": rotor, "ECG Start": ecg_start, "ECG End": ecg_end,
              "Y Constraint Check": constrain_check_y, "X Constraint Check": constrain_check_x,
              "Y Zero Check": zero_y_check, "X Zero Check": zero_x_check, "Y Loop": yloop, "X Loop": xloop,
              "Machine Learning Models": [args[1], args[2], args[3], args[4]]}

if save_data == 'y':
    with open('%s.p' % save_data_name, 'wb') as f:
        cPickle.dump(final_data, f)
