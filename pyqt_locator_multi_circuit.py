import pyqtgraph as pg
import sys
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from math import copysign
import time
import copy
from sklearn.externals import joblib
from random import randint, choice
import analysis_theano as at
from Functions import ani_convert, feature_extract_multi_test_rt, multi_feature_compile_rt, check_signs, check_bsign
import propagate_singlecircuit as ps

args = sys.argv

# Loading in Machine Learning models
#####################################
# y_classifier_full = joblib.load(args[1])
# y_class = joblib.load(args[2])
# x_classifier_full = joblib.load(args[3])
# x_class = joblib.load(args[4])
# vsign_check = np.load('/Users/williamcheng/AF-Clean/vsign_tensor.npy')
# hsign_check = np.load('/Users/williamcheng/AF-Clean/hsign_tensor.npy')
# axessign_check = np.load('/Users/williamcheng/AF-Clean/axessign_tensor.npy')

y_classifier_full = joblib.load('modeldump\models_sc\sc4k_yreg_byclass.pkl')
y_class = joblib.load('modeldump\models_sc\sc4k_xaxis_class.pkl')
x_classifier_full = joblib.load('modeldump\models_sc\sc4k_xreg_byclass.pkl')
x_class = joblib.load('modeldump\models_sc\sc4k_target_xaxisrestricted.pkl')
vsign_check = np.load('vsign_tensor.npy')
hsign_check = np.load('hsign_tensor.npy')
axessign_check = np.load('axessign_tensor.npy')


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
a.set_multi_circuit(np.ravel_multi_index([cp_y_pos,cp_x_pos],(200,200)),np.ravel_multi_index([cp_y_pos2,cp_x_pos2],(200,200)))
tissue_reset = False

# Initialising ECG recording (randomises the probe x,y position)
current_ecg_x_pos = randint(20, 179)
current_ecg_y_pos = randint(0, 199)
ecg_processing = at.ECG(centre=(current_ecg_y_pos, current_ecg_x_pos), m='g_single')

# Initialising the animation window
app = QtGui.QApplication([])
win = pg.GraphicsWindow(border=True)
win.show()
win.setWindowTitle('animation')
w1 = win.addLayout()
view = w1.addViewBox()
img = pg.ImageItem()
img.setLevels([0, 50])
label = pg.LabelItem(justify='right', border='w')
win.addItem(label)
view.addItem(img)
view.setRange(QtCore.QRectF(0, 0, 200, 200))


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
                return possible_points[0]
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
    binary = False
    x_jump = False
    b = check_bsign(total_sign[0][-2], total_sign[0][-1])
    h = total_sign_info[0][1]
    v = total_sign_info[0][0]
    if b != 0:
        binary = True
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
                print u, l, m, constrained_y
                if (u-m) * (m-l) >= 0:
                    print 'Special Constrain'
                    if p:
                        constrained_y[0] = m
                    else:
                        constrained_y[1] = m
                    break

        if h == -1:
            contrained_x[0] = current_x

        if h == 1:
            contrained_x[1] = current_x

    x_jump = True

    return contrained_x, constrained_y, binary, x_jump


def update_label_text(rotor_x_1, rotor_y_1, rotor_x_2, rotor_y_2,
                      ecg_x, ecg_y, ecg_num, prev_res, xaxis_const, yaxis_const, Pconst, nypos, nxpos, nyloop,
                      nxloop, nycons, nxcons, nyz, nxz):
    """

    :return:
    """
    text = """###################################<br>\n
              <br>\n
              Rotor 1 Position: (%s, %s)<br>\n
              Rotor 2 Position: (%s, %s)<br>\n
              Probe Position: (%s, %s)<br>\n
              <br>\n
              Previous Result: %s<br>\n
              Y Constraint: %s<br>\n
              X Constraint: %s<br>\n
              Perminant Constraint: %s<br>\n
              <br>\n
              Number of ECGs: %s<br>\n
              Number of Positive Y Classifications: %s<br>\n
              Number of Positive X Classifications: %s<br>\n
              Number of Y Loops: %s<br>\n
              Number of X Loops: %s<br>\n
              Number of Fully Constrained X Axis: %s<br>\n
              Number of Fully Constrained Y Axis: %s<br>\n
              Number of Y Zero Jumps: %s<br>\n
              Number of X Zero Jumps: %s<br>\n """ % (rotor_x_1, rotor_y_1, rotor_x_2, rotor_y_2, ecg_x, ecg_y,
                                                       prev_res, xaxis_const, yaxis_const, Pconst, ecg_num, nypos,
                                                       nxpos, nyloop, nxloop, nycons, nxcons, nyz, nxz)
    label.setText(text)


# Animation grid
animation_grid = np.zeros(a.shape)

# Constraint graphics
xUline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('c', width=4))
xLline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('b', width=4))
yUline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=4))
yLline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=4))
pUline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('m', width=4))
pLline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', width=4))
view.addItem(xUline, ignoreBounds=True)
view.addItem(xLline, ignoreBounds=True)
view.addItem(yUline, ignoreBounds=True)
view.addItem(yLline, ignoreBounds=True)
view.addItem(pUline, ignoreBounds=True)
view.addItem(pLline, ignoreBounds=True)

# Crosshair setup
vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2))
view.addItem(vLine, ignoreBounds=True)
view.addItem(hLine, ignoreBounds=True)

# time step
ptr1 = 0
ecg_count = 0
num_ypos_class = 0
num_xpos_class = 0
num_Yloops = 0
num_Xloops = 0
num_yconstraint = 0
num_xconstraint = 0
num_yzjump = 0
num_xzjump = 0
previousR = "None"
updateTime = ptime.time()
fps = 0
pred_rotor_x = []
pred_rotor_y = []

# Rotor state
rotors_found = 0

# estimated number of rotors (Needs a function to predict this value)
N_rotors = 1

# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 150

# Time before the ECG starts taking measurments (should it be just 400?)
n = 2
stability_time = n * process_length

# process list
process_list = []

prev_y_vector = None
prev_x_vector = None

# vector sign history
vsign_short_memory = []
hsign_short_memory = []

# History information
total_sign_info = []
vconsistent = []
hconsistent = []
bconsistent = []
x_history = []
y_history = []

special_state = False
binary_state = False
binary_jump = 0
jump_x = False

# Constrained y/x values
constrainedy = [None, None]
constrainedx = [20, 179]

# List of perminant constraints (gets reset when all rotors are found)
perminant_constraints = []

# Loop checking
y_short_memory = []
x_short_memory = []

# Setting measurment flag to False (ecg measurments start when flag is triggered).
ECG_start_flag = False

# Sate for pipe work.
state = 0

update_label_text(cp_x_pos, cp_y_pos, cp_x_pos2, cp_y_pos2, current_ecg_x_pos, current_ecg_y_pos, ecg_count, previousR,
                  constrainedy, constrainedx, perminant_constraints, num_ypos_class, num_xpos_class, num_Yloops,
                  num_Xloops, num_yconstraint, num_xconstraint, num_yzjump, num_xzjump)


# Updates the frames and goes through pipework for ECG processing and machine learning processes.
def update_data():
    global updateTime, fps, ptr1, process_list, ECG_start_flag, state, vsign_short_memory, constrainedy, constrainedx
    global current_ecg_y_pos, current_ecg_x_pos, y_short_memory, x_short_memory, ecg_count, previousR, prev_y_vector
    global prev_x_vector, hsign_short_memory, num_ypos_class, num_xpos_class, num_Yloops, num_Xloops, num_yconstraint
    global num_xconstraint, num_xzjump, num_yzjump, rotors_found, perminant_constraints, N_rotors, total_sign_info, vconsistent, hconsistent, bconsistent
    global x_history, y_history, special_state, binary_jump, binary_state, jump_x, special_vsign, special_y

    data = a.propagate(ecg=True)
    data = ani_convert(data, shape=a.shape, rp=a.rp, animation_grid=animation_grid)

    # Initial Crosshair drawing
    if ptr1 == 0:
        yUline.setPos(constrainedx[1])
        yLline.setPos(constrainedx[0])
        xUline.setPos(300)
        xLline.setPos(300)
        pUline.setPos(-300)
        pLline.setPos(-300)
        vLine.setPos(current_ecg_x_pos + 0.5)
        hLine.setPos(current_ecg_y_pos + 0.5)

    # If flag triggered, then start taking measurments.
    if ECG_start_flag:
        process_list.append(copy.deepcopy(data))

    ptr1 += 1

    # CONDITION TO START THE LOCATING PROCESS
    if ptr1 >= stability_time:
        if not ECG_start_flag:
            ECG_start_flag = True

        # CONDITION TO TAKE A MEASURMENT
        if ptr1 % process_length == 0 and ptr1 != stability_time:

            sample, signs = rt_ecg_gathering(process_list, sign_para='record_sign_plus')  # ECG Recording and feature gathering
            ecg_count += 1
            total_sign_info.append(signs)
            x_history.append(current_ecg_x_pos)
            y_history.append(current_ecg_y_pos)

            # LOOKING FOR THE ROTORS X AXIS
            if state == 0:
                sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                vsign = sample[0, :][-3]
                # first potential x constraint
                # sample_ = sample[0, :][0:-3].reshape(1, -1)  # Get sample without sign information.
                y_class_value = y_class.predict(sample)[0]
                y_probarg = reg_predictor(y_classifier_full.predict_proba(sample)[0, :])

                # POSITIVE CLASSIFICATION FOR Y
                if y_class_value == 1:
                    if special_state:
                        special_state = False
                    state = 1  # Change to state 1 for y axis regression/classification.
                    del y_short_memory
                    y_short_memory = []
                    del vsign_short_memory
                    vsign_short_memory = []
                    x_class_value = x_class.predict(sample)[0]
                    num_ypos_class += 1

                    # CHECKS IF ITS ON THE CORRECT Y AXIS
                    if x_class_value == 1:
                        previousR = "(%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos)
                        state = 0
                        ecg_count = 0
                        pred_rotor_y.append(current_ecg_y_pos)
                        pred_rotor_x.append(current_ecg_x_pos)

                        # ALL ROTORS ARE FOUND
                        if rotors_found == N_rotors:
                            current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            xUline.setPos(300)
                            xLline.setPos(300)
                            pUline.setPos(-300)
                            pLline.setPos(-300)
                            rotors_found = 0
                            perminant_constraints = []
                            binary_state = False
                            binary_jump = 0

                        # ONE OF THE ROTORS IS FOUND
                        else:
                            rotors_found += 1
                            upper = current_ecg_y_pos - 5
                            upper %= 200
                            lower = current_ecg_y_pos + 5
                            lower %= 200
                            perminant_constraints.append([lower, upper])

                            xUline.setPos(300)
                            xLline.setPos(300)
                            pUline.setPos(upper)
                            pLline.setPos(lower)
                            xvec, yvec = relative_vectors(x_history, y_history, current_ecg_x_pos, current_ecg_y_pos)
                            xvec = np.array(xvec)
                            yvec = np.array(yvec)

                            for i in range(len(x_history)):

                                vconsistent.append(check_signs(xvec[i],yvec[i],total_sign_info[i][0],vsign_check,thr = 0.05))
                                hconsistent.append(check_signs(xvec[i],yvec[i],total_sign_info[i][1],hsign_check,thr = 0.02))
                                bconsistent.append(check_bsign(total_sign_info[i][-2],total_sign_info[i][-1]))
                            vconsistent = np.array(vconsistent)[np.absolute(yvec) > 4]
                            hconsistent = np.array(hconsistent)[np.absolute(yvec) > 4]
                            bconsistent = np.array(bconsistent)[np.absolute(yvec) > 4]
                            total_sign_info = np.array(total_sign_info)
                            tsign = total_sign_info[np.absolute(yvec) > 4]
                            yvec_use = yvec[np.absolute(yvec) > 4]
                            xvec_use = xvec[np.absolute(yvec) > 4]
                            special_y = (yvec_use[vconsistent == False] + current_ecg_y_pos) % 200
                            special_vsign = tsign[:,0][vconsistent == False]

                            current_ecg_y_pos = (current_ecg_y_pos + 100) % 200
                            special_state = True

                        constrainedy = [None, None]
                        constrainedx = [20, 179]
                        yUline.setPos(constrainedx[1])
                        yLline.setPos(constrainedx[0])
                        num_xpos_class += 1
                        del y_short_memory
                        y_short_memory = []
                        del vsign_short_memory
                        vsign_short_memory = []
                        total_sign_info = []
                        vconsistent = []
                        hconsistent = []
                        bconsistent = []
                        x_history = []
                        y_history = []

                # NEGATIVE CLASSIFIACTION FOR Y
                if y_class_value == 0:
                    y_short_memory.append(current_ecg_y_pos)
                    vsign_short_memory.append(vsign)
                    if not special_state:
                        constrainedy, vsign_short_memory = constrained_finder(prev_y_vector, vsign_short_memory,
                                                                              current_ecg_y_pos, constrainedy, 'x',
                                                                              perminant_constraints)
                    if special_state:
                        constrainedx, constrainedy, binary_state, jump_x = special_constraint_finder(current_ecg_x_pos, current_ecg_y_pos, total_sign_info, constrainedy,
                                                                                             constrainedx, perminant_constraints,special_y,special_vsign)


                    # CONSTRAINED CONDITION FOR Y
                    if condistance(constrainedy) == 0:
                        state = 0
                        ecg_count = 0
                        num_yconstraint += 1
                        xUline.setPos(300)
                        xLline.setPos(300)
                        constrainedy = [None, None]
                        constrainedx = [20, 179]
                        if rotors_found > 0:
                            current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                        else:
                            current_ecg_y_pos = randint(0, 199)
                        current_ecg_x_pos = randint(20, 179)
                        yUline.setPos(constrainedx[1])
                        yLline.setPos(constrainedx[0])
                        del y_short_memory
                        y_short_memory = []
                        del vsign_short_memory
                        vsign_short_memory = []
                        total_sign_info = []
                        vconsistent = []
                        hconsistent = []
                        bconsistent = []
                        x_history = []
                        y_history = []

                    # MOVING THE PROBE IN THE Y AXIS
                    else:
                        if not binary_state:
                            likelyp = prediction(y_probarg, vector_constraint=vecdistance(current_ecg_y_pos,
                                                                                          constrainedy), axis='x')
                            y_vector = y_classifier_full.classes_[likelyp]

                            if np.abs(y_vector) > 45 and special_state:
                                print y_vector
                                y_vector = 45 * copysign(1, y_vector)
                                special_state = False
                                print y_vector

                        if binary_state:
                            print "Binary If"
                            print binary_jump
                            if binary_jump == 1:
                                print current_ecg_y_pos
                                y_vector = -100
                                binary_jump = 0
                                binary_state = False
                            else:
                                y_vector = 50
                                binary_jump += 1

                        if jump_x:
                            h = total_sign_info[-1][1]
                            if np.abs(h) > 0.4:
                                if h > 0:
                                    current_ecg_x_pos = int((current_ecg_x_pos + 20) / 2.)
                                else:
                                    current_ecg_x_pos = int((current_ecg_x_pos + 179) / 2.)

                            else:
                                xr = 179 - current_ecg_x_pos
                                xl = current_ecg_x_pos - 20
                                if xr >= xl:
                                    current_ecg_x_pos = int((current_ecg_x_pos + 179) / 2.)
                                else:
                                    current_ecg_x_pos = int((current_ecg_x_pos + 20) / 2.)
                            jump_x = False

                        # IF THE PREDICTED Y JUMP IS ZERO
                        if y_vector == 0:
                            previousR = "(%s, %s) (0 Y Jump)" % (current_ecg_x_pos, current_ecg_y_pos)
                            state = 0
                            ecg_count = 0
                            num_yzjump += 1
                            xUline.setPos(300)
                            xLline.setPos(300)
                            constrainedy = [None, None]
                            constrainedx = [20, 179]
                            if rotors_found > 0:
                                current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                            else:
                                current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            yUline.setPos(constrainedx[1])
                            yLline.setPos(constrainedx[0])
                            del y_short_memory
                            y_short_memory = []
                            del vsign_short_memory
                            vsign_short_memory = []

                        prev_y_vector = y_vector
                        current_ecg_y_pos -= y_vector

                        # WRAPPING AROUND THE BOUNDRY CONDITION
                        if current_ecg_y_pos > 199 or current_ecg_y_pos < 0:
                            current_ecg_y_pos %= 200

                        # Y LOOP FORM CHECK
                        if current_ecg_y_pos in y_short_memory:
                            previousR = 'Y LOOP'
                            state = 0
                            ecg_count = 0
                            num_Yloops += 1
                            xUline.setPos(300)
                            xLline.setPos(300)
                            constrainedy = [None, None]
                            constrainedx = [20, 179]
                            if rotors_found > 0:
                                current_ecg_y_pos = (current_ecg_y_pos + 100)%200
                            else:
                                current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            yUline.setPos(constrainedx[1])
                            yLline.setPos(constrainedx[0])
                            del y_short_memory
                            y_short_memory = []
                            del vsign_short_memory
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
                # sample_ = sample[0, :][0:-3].reshape(1, -1)  # Takes a sample without sign information

                x_probarg = reg_predictor(x_classifier_full.predict_proba(sample)[0, :])  # Prob map
                x_class_value = x_class.predict(sample)[0]

                # POSITIVE CLASSIFICATION FOR X
                if x_class_value == 1:
                    previousR = "(%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos)
                    state = 0
                    ecg_count = 0
                    num_xpos_class += 1
                    pred_rotor_y.append(current_ecg_y_pos)
                    pred_rotor_x.append(current_ecg_x_pos)

                    if rotors_found == N_rotors:
                        xUline.setPos(300)
                        xLline.setPos(300)
                        pUline.setPos(-300)
                        pLline.setPos(-300)
                        current_ecg_x_pos = randint(20, 179)
                        current_ecg_y_pos = randint(0, 199)
                        rotors_found = 0
                        perminant_constraints = []

                    else:
                        rotors_found += 1
                        upper = current_ecg_y_pos - 5
                        upper %= 200
                        lower = current_ecg_y_pos + 5
                        lower %= 200
                        perminant_constraints.append([lower, upper])

                        xUline.setPos(300)
                        xLline.setPos(300)
                        pUline.setPos(upper)
                        pLline.setPos(lower)
                        xvec, yvec = relative_vectors(x_history, y_history, current_ecg_x_pos, current_ecg_y_pos)
                        xvec = np.array(xvec)
                        yvec = np.array(yvec)

                        for i in range(len(x_history)):

                            vconsistent.append(check_signs(xvec[i],yvec[i],total_sign_info[i][0],vsign_check,thr = 0.05))
                            hconsistent.append(check_signs(xvec[i],yvec[i],total_sign_info[i][1],hsign_check,thr = 0.02))
                            bconsistent.append(check_bsign(total_sign_info[i][-2],total_sign_info[i][-1]))
                        vconsistent = np.array(vconsistent)[np.absolute(yvec) > 4]
                        hconsistent = np.array(hconsistent)[np.absolute(yvec) > 4]
                        bconsistent = np.array(bconsistent)[np.absolute(yvec) > 4]
                        total_sign_info = np.array(total_sign_info)
                        tsign = total_sign_info[np.absolute(yvec) > 4]
                        yvec_use = yvec[np.absolute(yvec) > 4]
                        xvec_use = xvec[np.absolute(yvec) > 4]
                        special_y = (yvec_use[vconsistent == False] + current_ecg_y_pos) % 200
                        special_vsign = tsign[:,0][vconsistent == False]

                        current_ecg_y_pos = (current_ecg_y_pos + 100) % 200
                        special_state = True

                    constrainedy = [None, None]
                    constrainedx = [20, 179]
                    yUline.setPos(constrainedx[1])
                    yLline.setPos(constrainedx[0])
                    del x_short_memory
                    x_short_memory = []
                    del hsign_short_memory
                    hsign_short_memory = []
                    total_sign_info = []
                    vconsistent = []
                    hconsistent = []
                    bconsistent = []
                    x_history = []
                    y_history = []

                # NEGATIVE CLASSIFICATION FOR X
                if x_class_value == 0:
                    x_short_memory.append(current_ecg_x_pos)
                    hsign_short_memory.append(hsign)
                    constrainedx, hsign_short_memory = constrained_finder(prev_x_vector, hsign_short_memory,
                                                                          current_ecg_x_pos, constrainedx, 'y',
                                                                          perminant_constraints)

                    # CONSTRAINED CONDITION FOR X
                    if condistance(constrainedx) == 0:  # Row is constrained to be have distance 1, take position.
                        previousR = "(%s, %s) (Constrained)" % (current_ecg_x_pos, current_ecg_y_pos)
                        state = 0
                        ecg_count = 0
                        num_xconstraint += 1
                        xUline.setPos(300)
                        xLline.setPos(300)
                        constrainedy = [None, None]
                        constrainedx = [20, 179]
                        if rotors_found > 0:
                            current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                        else:
                            current_ecg_y_pos = randint(0, 199)
                        current_ecg_x_pos = randint(20, 179)
                        yUline.setPos(constrainedx[1])
                        yLline.setPos(constrainedx[0])
                        del x_short_memory
                        x_short_memory = []
                        del hsign_short_memory
                        hsign_short_memory = []
                        total_sign_info = []
                        x_history = []
                        y_history = []
                        vconsistent = []
                        hconsistent = []
                        bconsistent = []

                    # MOVING THE PROBE IN THE X AXIS
                    else:
                        likelyp = prediction(x_probarg, vector_constraint=vecdistance(current_ecg_x_pos,
                                                                                      constrainedx), axis='y')
                        x_vector = x_classifier_full.classes_[likelyp]

                        # IF THE PREDICTED JUMP IS ZERO
                        if x_vector == 0:  # If the predicted X jump is 0
                            previousR = "(%s, %s) (0 X Jump)" % (current_ecg_x_pos, current_ecg_y_pos)
                            state = 0
                            ecg_count = 0
                            num_xzjump += 1
                            xUline.setPos(300)
                            xLline.setPos(300)
                            constrainedy = [None, None]
                            constrainedx = [20, 179]
                            if rotors_found > 0:
                                current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                            else:
                                current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            yUline.setPos(constrainedx[1])
                            yLline.setPos(constrainedx[0])
                            del x_short_memory
                            x_short_memory = []
                            del hsign_short_memory
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
                            previousR = '(X LOOP, %s)' % current_ecg_y_pos
                            state = 0
                            ecg_count = 0
                            num_Xloops += 1
                            xUline.setPos(300)
                            xLline.setPos(300)
                            constrainedy = [None, None]
                            constrainedx = [20, 179]
                            if rotors_found > 0:
                                lower = perminant_constraints[0][0]
                                upper = perminant_constraints[0][1]
                                # current_ecg_y_pos = choice(conposition(lower, upper))
                                # current_ecg_y_pos = (current_ecg_y_pos + 100) % 200
                                current_ecg_y_pos = (pred_rotor_y[-1] + 100) % 200
                            else:
                                current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            yUline.setPos(constrainedx[1])
                            yLline.setPos(constrainedx[0])
                            del x_short_memory
                            x_short_memory = []
                            del hsign_short_memory
                            hsign_short_memory = []
                            total_sign_info = []
                            x_history = []
                            y_history = []
                            vconsistent = []
                            hconsistent = []
                            bconsistent = []

            # UPDATING LINES AND PREPARING FOR NEW MEASURMENT
            ecg_processing.reset_singlegrid((current_ecg_y_pos, current_ecg_x_pos))
            if constrainedy[0] is not None:
                xLline.setPos(constrainedy[0])
            if constrainedy[1] is not None:
                xUline.setPos(constrainedy[1])
            yLline.setPos(constrainedx[0])
            yUline.setPos(constrainedx[1])
            vLine.setPos(current_ecg_x_pos + 0.5)
            hLine.setPos(current_ecg_y_pos + 0.5)

            del process_list
            process_list = []
            update_label_text(cp_x_pos, cp_y_pos, cp_x_pos2, cp_y_pos2, current_ecg_x_pos, current_ecg_y_pos, ecg_count,
                              previousR, constrainedy, constrainedx, perminant_constraints, num_ypos_class,
                              num_xpos_class, num_Yloops, num_Xloops, num_yconstraint, num_xconstraint, num_yzjump,
                              num_xzjump)

    # time.sleep(1/120.)  # gives more stable fps.
    img.setImage(data.T)  # puts animation grid on image.

    # Stuff to do with time and fps.
    QtCore.QTimer.singleShot(1, update_data)
    now = ptime.time()
    fps2 = 1.0 / (now - updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    # print "%0.1f fps" % fps

# updates the animation frames.
update_data()

# Need this at the end for some reason...
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
