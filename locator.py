"""
Runs the locator algorithm but without the animation (animation in pyqt_locator.py)
"""

import numpy as np
import copy
from sklearn.externals import joblib
from random import randint
import analysis_theano as at
from Functions import ani_convert, feature_extract_multi_test_rt, multi_feature_compile_rt
import propagate_singlesource as ps

# Loading in Machine Learning models
#####################################
y_regress = joblib.load('y_regress_rt_4.pkl')
y_estimator = joblib.load('y_class_rt_1.pkl')
x_regress = joblib.load('x_regress_rt_2.pkl')
x_class = joblib.load('x_classifier_rt_1.pkl')
#####################################

# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 360

# Time before the ECG starts taking measurments (should it be just 400?)
n = 1
stability_time = n * process_length


def rt_ecg_gathering(ecg_list):
    """
    Records the ECGS, Gathers the features and compiles them.
    :param ecg_list: Raw data from animation_grid (t, (x,y))
    :return: (441,) array of feature data.
    """
    voltages = ecg_processing.solve(np.array(ecg_list).astype('float32'))

    # Putting all 9 ecg features into (9,21) array
    uncompiled_features = []
    for i in range(9):
        uncompiled_features.append(feature_extract_multi_test_rt(i, voltages))
    compiled_features = multi_feature_compile_rt(np.array(uncompiled_features))
    return compiled_features

# Number of rotors to find
number_of_rotors = 100
# If this number of steps is reached before rotor is found, then give up and move on.
upper_limit = 50

# Lists for recording data produced by algorithm
ecg_counter = [0]*number_of_rotors
ecg_start = [0]* number_of_rotors
ecg_end = [0]*number_of_rotors
rotor = [0]*number_of_rotors

for i in range(number_of_rotors):

    # Initialising the Heart structure
    a = ps.Heart(nu=0.2, delta=0.0, fakedata=True)
    # Randomises the rotor x,y position
    cp_x_pos = randint(0, 199)
    cp_y_pos = randint(0, 199)
    a.set_pulse(60, [[cp_y_pos], [cp_x_pos]])

    # Initialising ECG recording (randomises the probe x,y position)
    current_ecg_x_pos = randint(3, 196)
    current_ecg_y_pos = randint(3, 196)
    ecg_processing = at.ECG(centre=(current_ecg_y_pos, current_ecg_x_pos), m='g_single')

    # Animation grids
    animation_grid = np.zeros(a.shape)

    # reset time step counter
    ptr1 = 0

    # process list
    process_list = []

    # Setting measurment flag to False (ecg measurments start when flag is triggered).
    ECG_start_flag = False
    # To keep the system propogating until the rotor is found or limit is reached.
    ECG_located_flag = False

    # Sate for pipe work.
    state = 0

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
                print "Starting Measurment Process"
                ECG_start_flag = True
            if ptr1 % process_length == 0 and ptr1 != stability_time:



