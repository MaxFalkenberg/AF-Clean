import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time
import copy
from random import randint
import analysis_theano as at
from Functions import ani_convert
import propagate_singlesource as ps

# Loading in Machine Learning models
#####################################

#####################################

# Initialising the Heart structure
a = ps.Heart(nu=0.2, delta=0.05, fakedata=True)
#randomises the x/y position
cp_x_pos = randint(0, 199)
cp_y_pos = randint(0, 199)
a.set_pulse(60, [[cp_y_pos], [cp_x_pos]])

# Initialising ECG recording (randomises the x/y position)
ecg_x_pos = randint(0, 199)
ecg_y_pos = randint(0, 199)
r = at.ECG(center)

# Initialising the animation window
app = QtGui.QApplication([])
win = pg.GraphicsWindow()
win.show()
win.setWindowTitle('animation')
view = win.addPlot()
img = pg.ImageItem(border='w')
label = pg.TextItem()
view.addItem(label)
view.hideAxis('left')
view.hideAxis('bottom')
view.addItem(img)
view.setRange(QtCore.QRectF(0, -20, 200, 220))

#Animation grid.
animation_grid = np.zeros(a.shape)

# time step
ptr1 = 0
updateTime = ptime.time()
fps = 0
# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 120
# process list
process_list = []


def update_data():
    global updateTime, fps, ptr1, process_list
    data = a.propagate(ecg=True)
    data = ani_convert(data, shape=a.shape, rp=a.rp, animation_grid=animation_grid)
    process_list.append(copy.deepcopy(data))
    ptr1 += 1
    if ptr1 % process_length == 0:
        # needs a state variable to decide what to do
        # DO STUFF
        del process_list
        process_list = []
    time.sleep(1/120.)  # gives larger more stable fps.
    img.setImage(data.T, levels=(0, 50))

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
