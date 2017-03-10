import pyqtgraph as pg
import sys
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from math import copysign
import time
import copy
from sklearn.externals import joblib
from random import randint
import analysis_theano as at
from Functions import ani_convert, feature_extract_multi_test_rt, multi_feature_compile_rt
import propagate_singlesource as ps

args = sys.argv

# Loading in Machine Learning models
#####################################
y_classifier_full = joblib.load(args[1])
y_class = joblib.load(args[2])
x_classifier_full = joblib.load(args[3])
x_class = joblib.load(args[4])
#####################################

# Initialising the Heart structure
a = ps.Heart(nu=0.2, delta=0.0, fakedata=True)
# Randomises the rotor x,y position
cp_x_pos = randint(30, 169)
cp_y_pos = randint(0, 199)
a.set_pulse(60, [[cp_y_pos], [cp_x_pos]])
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
    compiled_features = multi_feature_compile_rt(np.array(uncompiled_features), sign=sign_para)
    return compiled_features


def movingaverage(values, weight):
    sma = np.convolve(values, np.ones((weight,)) / weight, mode='same')
    return sma


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
    # if current_pos == 0:    #  This might have caused the error seen.
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
                if prev_vector < -3 and constrained_[0] is not None:
                    constrained_[0] += 3
                    constrained_[0] %= 200
            constrained_[1] = current_ecg_pos_

        if prev_vector > 0 and vsign_diff == -2:  # Lower Constraint
            if axis == 'x':
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
                if axis == 'x':
                    if prev_vector > 3:
                        constrained_[1] -= 3
                        constrained_[1] %= 200
            if condistance(constrained_) > condistance([constrained_[0], current_ecg_pos_]):
                constrained_[1] = current_ecg_pos_

        if prev_vector < 0 and vsign_diff == 0:  # Potential updating of lower constraint
            if constrained_[1] is None:
                constrained_[0] = current_ecg_pos_
                if axis == 'x':
                    if prev_vector < -3:
                        constrained_[0] += 3
                        constrained_[0] %= 200
            if condistance(constrained_) > condistance([current_ecg_pos_, constrained_[1]]):
                constrained_[0] = current_ecg_pos_

    return constrained_, sign_short_memory_


def update_label_text(rotor_x, rotor_y, ecg_x, ecg_y, ecg_num, prev_res, xaxis_const, yaxis_const):
    """

    :return:
    """
    text = """############################################<br>\n
              <br>\n
              Rotor Position: (%s, %s)<br>\n
              Probe Position: (%s, %s)<br>\n
              <br>\n
              Number of ECGs: %s<br>\n
              Previous Result: %s<br>\n
              Y Constraint: %s<br>\n
              X Constraint: %s""" % (rotor_x, rotor_y, ecg_x, ecg_y, ecg_num, prev_res,
                                     xaxis_const, yaxis_const)
    label.setText(text)


# Animation grid
animation_grid = np.zeros(a.shape)

# Constraint graphics
xUline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('c', width=4))
xLline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('b', width=4))
yUline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=4))
yLline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=4))
view.addItem(xUline, ignoreBounds=True)
view.addItem(xLline, ignoreBounds=True)
view.addItem(yUline, ignoreBounds=True)
view.addItem(yLline, ignoreBounds=True)

# Crosshair setup
vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2))
view.addItem(vLine, ignoreBounds=True)
view.addItem(hLine, ignoreBounds=True)

# time step
ptr1 = 0
ecg_count = 0
previousR = "None"
updateTime = ptime.time()
fps = 0

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

# Constrained y/x values
constrainedy = [None, None]
constrainedx = [20, 179]

# Loop checking
y_short_memory = []
x_short_memory = []

# Setting measurment flag to False (ecg measurments start when flag is triggered).
ECG_start_flag = False

# Sate for pipe work.
state = 0

update_label_text(cp_x_pos, cp_y_pos, current_ecg_x_pos, current_ecg_y_pos, ecg_count, previousR,
                  constrainedy, constrainedx)


# Updates the frames and goes through pipework for ECG processing and machine learning processes.
def update_data():
    global updateTime, fps, ptr1, process_list, ECG_start_flag, state, vsign_short_memory, constrainedy, constrainedx
    global current_ecg_y_pos, current_ecg_x_pos, y_short_memory, x_short_memory, ecg_count, previousR, prev_y_vector
    global prev_x_vector, hsign_short_memory

    data = a.propagate(ecg=True)
    data = ani_convert(data, shape=a.shape, rp=a.rp, animation_grid=animation_grid)

    # Initial Crosshair drawing
    if ptr1 == 0:
        yUline.setPos(constrainedx[1])
        yLline.setPos(constrainedx[0])
        xUline.setPos(300)
        xLline.setPos(300)
        vLine.setPos(current_ecg_x_pos + 0.5)
        hLine.setPos(current_ecg_y_pos + 0.5)

    # If flag triggered, then start taking measurments.
    if ECG_start_flag:
        process_list.append(copy.deepcopy(data))

    ptr1 += 1

    if ptr1 >= stability_time:
        if not ECG_start_flag:
            ECG_start_flag = True

        if ptr1 % process_length == 0 and ptr1 != stability_time:

            sample = rt_ecg_gathering(process_list, sign_para=args[5])  # ECG Recording and feature gathering
            ecg_count += 1

            if state == 0:
                sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                vsign = sample[0, :][-3]
                # first potential x constraint
                # sample_ = sample[0, :][0:-3].reshape(1, -1)  # Get sample without sign information.
                y_class_value = y_class.predict(sample)[0]
                y_probarg = movingaverage(y_classifier_full.predict_proba(sample)[0, :], 10)

                if y_class_value == 1:
                    state = 1  # Change to state 1 for y axis regression/classification.
                    del y_short_memory
                    y_short_memory = []
                    del vsign_short_memory
                    vsign_short_memory = []
                    x_class_value = x_class.predict(sample)[0]

                    if x_class_value == 1:
                        previousR = "(%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos)
                        del y_short_memory
                        y_short_memory = []
                        del vsign_short_memory
                        vsign_short_memory = []
                        del constrainedy
                        constrainedy = [None, None]
                        current_ecg_y_pos = randint(20, 179)  # reseting the process.
                        current_ecg_x_pos = randint(20, 179)
                        state = 0
                        ecg_count = 0
                        yUline.setPos(constrainedx[1])
                        yLline.setPos(constrainedx[0])
                        xUline.setPos(300)
                        xLline.setPos(300)

                if y_class_value == 0:
                    y_short_memory.append(current_ecg_y_pos)
                    vsign_short_memory.append(vsign)
                    constrainedy, vsign_short_memory = constrained_finder(prev_y_vector, vsign_short_memory,
                                                                          current_ecg_y_pos, constrainedy, axis='x')
                    if condistance(constrainedy) == 1:
                        state = 1
                        x_class_value = x_class.predict(sample)[0]

                        if x_class_value == 1:
                            previousR = "(%s, %s) (Constrained)" % (current_ecg_x_pos, current_ecg_y_pos)
                            # reseting the process.
                            del y_short_memory
                            y_short_memory = []
                            del vsign_short_memory
                            vsign_short_memory = []
                            del constrainedy
                            constrainedy = [None, None]
                            current_ecg_y_pos = randint(0, 199)
                            current_ecg_x_pos = randint(20, 179)
                            state = 0
                            ecg_count = 0
                            yUline.setPos(constrainedx[1])
                            yLline.setPos(constrainedx[0])
                            xUline.setPos(300)
                            xLline.setPos(300)

                    else:
                        likelyp = prediction(y_probarg, vector_constraint=vecdistance(current_ecg_y_pos,
                                                                                      constrainedy), axis='x')
                        y_vector = y_classifier_full.classes_[likelyp]

                        if y_vector == 0:
                            state = 1  # Change to state 1 for y axis regression/classification.
                            del y_short_memory
                            y_short_memory = []
                            del vsign_short_memory
                            vsign_short_memory = []
                            x_class_value = x_class.predict(sample)[0]

                            if x_class_value == 1:
                                previousR = "(%s, %s) (0 Y Jump)" % (current_ecg_x_pos, current_ecg_y_pos)
                                del y_short_memory
                                y_short_memory = []
                                del vsign_short_memory
                                vsign_short_memory = []
                                del constrainedy
                                constrainedy = [None, None]
                                current_ecg_y_pos = randint(20, 179)  # reseting the process.
                                current_ecg_x_pos = randint(20, 179)
                                state = 0
                                ecg_count = 0
                                yUline.setPos(constrainedx[1])
                                yLline.setPos(constrainedx[0])
                                xUline.setPos(300)
                                xLline.setPos(300)  # If it predicts a jump of 0, then take the position it's on.

                        prev_y_vector = y_vector
                        current_ecg_y_pos -= y_vector

                        if current_ecg_y_pos > 199 or current_ecg_y_pos < 0:
                            current_ecg_y_pos %= 200

            if state == 1:
                sample = sample.reshape(1, -1)  # Get deprication warning if this is not done.
                hsign = sample[0, :][-2]  # Gets the h sign
                # sample_ = sample[0, :][0:-3].reshape(1, -1)  # Takes a sample without sign information
                x_probarg = movingaverage(x_classifier_full.predict_proba(sample)[0, :], 10)  # Prob map
                x_class_value = x_class.predict(sample)[0]

                if x_class_value == 1:
                    previousR = "(%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos)
                    current_ecg_x_pos = randint(20, 179)
                    current_ecg_y_pos = randint(0, 199)
                    state = 0
                    ecg_count = 0
                    del x_short_memory
                    x_short_memory = []
                    del hsign_short_memory
                    hsign_short_memory = []
                    del constrainedx
                    constrainedx = [20, 179]
                    del constrainedy
                    constrainedy = [None, None]
                    yUline.setPos(constrainedx[1])
                    yLline.setPos(constrainedx[0])
                    xUline.setPos(300)
                    xLline.setPos(300)

                if x_class_value == 0:
                    x_short_memory.append(current_ecg_x_pos)
                    hsign_short_memory.append(hsign)
                    constrainedx, hsign_short_memory = constrained_finder(prev_x_vector, hsign_short_memory,
                                                                          current_ecg_x_pos, constrainedx, axis='y')

                    if condistance(constrainedx) == 1:  # Row is constrained to be have distance 1, take position.
                        previousR = "(%s, %s) (Constrained)" % (current_ecg_x_pos, current_ecg_y_pos)
                        state = 1
                        del x_short_memory
                        x_short_memory = []
                        del hsign_short_memory
                        hsign_short_memory = []
                        del constrainedx
                        constrainedx = [20, 179]
                        del constrainedy
                        constrainedy = [None, None]
                        current_ecg_x_pos = randint(20, 179)
                        current_ecg_y_pos = randint(0, 199)
                        state = 0
                        ecg_count = 0
                        yUline.setPos(constrainedx[1])
                        yLline.setPos(constrainedx[0])
                        xUline.setPos(300)
                        xLline.setPos(300)

                    else:
                        likelyp = prediction(x_probarg, vector_constraint=vecdistance(current_ecg_x_pos,
                                                                                      constrainedx), axis='y')
                        x_vector = x_classifier_full.classes_[likelyp]

                        if x_vector == 0:  # If the predicted X jump is 0
                            previousR = "(%s, %s) (0 X Jump)" % (current_ecg_x_pos, current_ecg_y_pos)
                            current_ecg_x_pos = randint(20, 179)
                            current_ecg_y_pos = randint(0, 199)
                            state = 0
                            ecg_count = 0
                            del x_short_memory
                            x_short_memory = []
                            del hsign_short_memory
                            hsign_short_memory = []
                            del constrainedx
                            constrainedx = [20, 179]
                            del constrainedy
                            constrainedy = [None, None]
                            yUline.setPos(constrainedx[1])
                            yLline.setPos(constrainedx[0])
                            xUline.setPos(300)
                            xLline.setPos(300)

                        prev_x_vector = x_vector
                        current_ecg_x_pos -= x_vector

                        if current_ecg_x_pos > 199 or current_ecg_x_pos < 0:
                            current_ecg_x_pos %= 200

                        if current_ecg_x_pos in x_short_memory:
                            previousR = '(X LOOP, %s)' % current_ecg_y_pos
                            state = 1
                            del x_short_memory
                            x_short_memory = []
                            del hsign_short_memory
                            hsign_short_memory = []
                            del constrainedx
                            constrainedx = [20, 179]
                            del constrainedy
                            constrainedy = [None, None]
                            current_ecg_x_pos = randint(20, 179)
                            current_ecg_y_pos = randint(0, 199)
                            state = 0
                            ecg_count = 0
                            yUline.setPos(constrainedx[1])
                            yLline.setPos(constrainedx[0])
                            xUline.setPos(300)
                            xLline.setPos(300)

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
            update_label_text(cp_x_pos, cp_y_pos, current_ecg_x_pos, current_ecg_y_pos, ecg_count, previousR,
                              constrainedy, constrainedx)

    time.sleep(1/120.)  # gives more stable fps.
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
