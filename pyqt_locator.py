import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time
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
#####################################

# Initialising the Heart structure
a = ps.Heart(nu=0.2, delta=0.0, fakedata=True)
# Randomises the rotor x,y position
cp_x_pos = randint(0, 199)
cp_y_pos = randint(0, 199)
a.set_pulse(60, [[cp_y_pos], [cp_x_pos]])

print "Ectopic Beat position: (%s, %s)" % (cp_x_pos, cp_y_pos)

# Initialising ECG recording (randomises the probe x,y position)
current_ecg_x_pos = randint(3, 196)
current_ecg_y_pos = randint(3, 196)
ecg_processing = at.ECG(centre=(current_ecg_y_pos, current_ecg_x_pos), m='g_single')

print "Initial ECG Probe position: (%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos)

# Initialising the animation window
app = QtGui.QApplication([])
win = pg.GraphicsWindow()
win.show()
win.setWindowTitle('animation')
view = win.addPlot()
img = pg.ImageItem(border='w')
img.setLevels([0, 50])
label = pg.TextItem()
view.addItem(label)
view.hideAxis('left')
view.hideAxis('bottom')
view.addItem(img)
view.setRange(QtCore.QRectF(0, -20, 200, 220))

# Initial probe and rotor position label.
label.setText("ECG Position: (%s, %s), Rotor Position: (%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos,
                                                                    cp_x_pos, cp_y_pos))

# Animation grids
animation_grid = np.zeros(a.shape)

# Crosshair setup
vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2))
view.addItem(vLine, ignoreBounds=True)
view.addItem(hLine, ignoreBounds=True)

vb = view.vb

# time step
ptr1 = 0
updateTime = ptime.time()
fps = 0

# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 360

# Time before the ECG starts taking measurments (should it be just 400?)
n = 1
stability_time = n * process_length

# process list
process_list = []

# Loop checking
y_short_memory = []

# Setting measurment flag to False (ecg measurments start when flag is triggered).
ECG_start_flag = False

# Sate for pipe work.
state = 0

"""
State 0 - always measure and process ECG
        - test y regression
        - if np.abs(y vector) < 3:
            keep features for further processing.
            state = 1
          if np.abs(y vector) >= 3:
            move probe by y vector
            state = 0
"""

# Threshold for y regression test (currently not used.)
y_regress_treshold = 3


def rt_ecg_gathering(process_list):
    """
    Records the ECGS, Gathers the features and compiles them.
    :param process_list: Raw data from animation_grid (t, (x,y))
    :return: (441,) array of feature data.
    """
    voltages = ecg_processing.solve(np.array(process_list).astype('float32'))

    # Putting all 9 ecg features into (9,21) array
    uncompiled_features = []
    for i in range(9):
        uncompiled_features.append(feature_extract_multi_test_rt(i, voltages))
    compiled_features = multi_feature_compile_rt(np.array(uncompiled_features))
    return compiled_features


# Updates the frames and goes through pipework for ECG processing and machine learning processes.
def update_data():
    global updateTime, fps, ptr1, process_list, ECG_start_flag, state, y_regress_treshold
    global current_ecg_y_pos, current_ecg_x_pos, y_short_memory

    data = a.propagate(ecg=True)
    data = ani_convert(data, shape=a.shape, rp=a.rp, animation_grid=animation_grid)

    # Initial Crosshair drawing
    if ptr1 == 0:
        vLine.setPos(current_ecg_x_pos + 0.5)
        hLine.setPos(current_ecg_y_pos + 0.5)

    # If flag triggered, then start taking measurments.
    if ECG_start_flag:
        process_list.append(copy.deepcopy(data))

    ptr1 += 1

    if ptr1 >= stability_time:
        if not ECG_start_flag:
            print "Starting Measurment Process"
            ECG_start_flag = True
        if ptr1 % process_length == 0 and ptr1 != stability_time:

            if state == 0:
                # ECG Recording and feature gathering
                sample = rt_ecg_gathering(process_list)
                # Get deprication warning if this is not done.
                sample = sample.reshape(1,-1)

                y_class = y_estimator.predict(sample)[0]
                print "Y classification: %s" % y_class
                y_vector = int(y_regress.predict(sample)[0])
                print "Y Vector prediction: %s" % y_vector

                if y_class == 1:
                    print "Found X-Axis"
                    # Temporary
                    del y_short_memory
                    y_short_memory = []
                    current_ecg_y_pos = randint(3, 196)
                    current_ecg_x_pos = randint(3, 196)

                if y_class == 0:
                    y_short_memory.append(current_ecg_y_pos)
                    current_ecg_y_pos -= y_vector
                    if current_ecg_y_pos > 200 or current_ecg_y_pos < 0:
                        current_ecg_y_pos %= 200
                    if current_ecg_y_pos in y_short_memory:
                        print "Entered Loop"
                        print "Loop: %s" % y_short_memory
                        loop_average = int((float(sum(y_short_memory))/len(y_short_memory)))
                        print "Loop Average: %s" % loop_average
                        del y_short_memory
                        y_short_memory = []
                        current_ecg_y_pos = loop_average
                        # current_ecg_x_pos =

            print "New ECG Probe position: (%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos)
            print '\n'
            ecg_processing.reset_singlegrid((current_ecg_y_pos, current_ecg_x_pos))
            vLine.setPos(current_ecg_x_pos + 0.5)
            hLine.setPos(current_ecg_y_pos + 0.5)
            del process_list
            process_list = []
            label.setText("ECG Position: (%s, %s), Rotor Position: (%s, %s)" % (current_ecg_x_pos, current_ecg_y_pos,
                                                                                cp_x_pos, cp_y_pos))

    # gives more stable fps.
    # time.sleep(1/120.)
    # puts animation grid on image.
    img.setImage(data.T)

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
