import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
import pandas as pd
import numpy as np
import time
import copy
from random import randint
import analysis_theano as at
from Functions import ani_convert, feature_extract_multi_test_rt
import propagate_singlesource as ps

# Loading in Machine Learning models
#####################################

#####################################

# Initialising the Heart structure
a = ps.Heart(nu=0.2, delta=0.05, fakedata=True)
# Randomises the x/y position
cp_x_pos = randint(0, 199)
cp_y_pos = randint(0, 199)
a.set_pulse(60, [[cp_y_pos], [cp_x_pos]])

print "Ectopic Beat position: (%s, %s)" % (cp_x_pos, cp_y_pos)

# Initialising ECG recording (randomises the x,y position)
ecg_x_pos = randint(0, 199)
ecg_y_pos = randint(0, 199)
ecg_processing = at.ECG(centre=(ecg_y_pos, ecg_x_pos), m='g_single')

print "Initial ECG Probe position: (%s, %s)" % (ecg_x_pos, ecg_y_pos)

# Initialising the animation window
app = QtGui.QApplication([])
win = pg.GraphicsWindow()
win.show()
win.setWindowTitle('animation')
view = win.addPlot()
img = pg.ImageItem(border='w')
# img.setLookupTable(lut)
img.setLevels([0, 50])
label = pg.TextItem()
view.addItem(label)
view.hideAxis('left')
view.hideAxis('bottom')
view.addItem(img)
view.setRange(QtCore.QRectF(0, 0, 200, 200))

# Animation grids
animation_grid = np.zeros(a.shape)

# Crosshair setup
vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2))
view.addItem(vLine, ignoreBounds=True)
view.addItem(hLine, ignoreBounds=True)

# time step
ptr1 = 0
updateTime = ptime.time()
fps = 0

# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 480

# Time before the ECG starts taking measurments (should it be just 400?)
n = 1
stability_time = n * process_length

# process list
process_list = []

ECG_start_flag = False


def update_data():
    global updateTime, fps, ptr1, process_list, ECG_start_flag
    data = a.propagate(ecg=True)
    data = ani_convert(data, shape=a.shape, rp=a.rp, animation_grid=animation_grid)

    # Initial Crosshair drawing
    if ptr1 == 0:
        vLine.setPos(ecg_x_pos + 0.5)
        hLine.setPos(ecg_y_pos + 0.5)

    # If flag triggered, then start taking measurments.
    if ECG_start_flag:
        process_list.append(copy.deepcopy(data))

    ptr1 += 1

    if ptr1 >= stability_time:
        if not ECG_start_flag:
            print "Starting Measurment Process"
            ECG_start_flag = True
        if ptr1 % process_length == 0 and ptr1 != stability_time:
            # needs a state variable to decide what to do
            print 'ECG OUTPUT'
            print ptr1
            voltages = ecg_processing.solve(np.array(process_list).astype('float32'))
            print voltages.shape

            uncompiled_features = []
            for i in range(9):
                uncompiled_features.append(feature_extract_multi_test_rt(i, voltages))
            print np.array(uncompiled_features).shape

            cuurent_ecg_x_pos = randint(0, 199)
            current_ecg_y_pos = randint(0, 199)
            ecg_processing.reset_singlegrid((current_ecg_y_pos, cuurent_ecg_x_pos))
            vLine.setPos(cuurent_ecg_x_pos + 0.5)
            hLine.setPos(current_ecg_y_pos + 0.5)
            del process_list
            process_list = []

    # gives larger more stable fps.
    time.sleep(1/120.)
    img.setImage(data.T)

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
