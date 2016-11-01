import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import basic_propagate as bc
import time
import analysis_theano as at

nu_value = float(raw_input('Choose a Nu value: '))
a = bc.Heart(nu_value)
#a = bc.fake_af()
a.set_pulse(220)
e = at.ECG_single(a.shape, 3)

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

animation_grid = np.zeros((200, 200))

win.nextRow()
p1 = win.addPlot()
p1.setYRange(-80, 40)
data1 = np.zeros(1000)
curve = p1.plot(data1, pen=pg.mkPen('w', width=2))
ptr1 = 0


def ani_convert(data):
    """
    Converts all the file data into arrays that can be animated.
    :return:
    """

    animation_grid[(animation_grid > 0) & (animation_grid <= 50)] -= 1
    if data == []:  # could use <if not individual_data.any():> but this is more readable.
        return animation_grid
    else:
        indices = np.unravel_index(data, a.shape)
        for ind in range(len(indices[0])):
            animation_grid[indices[0][ind]][indices[1][ind]] = 50
        return animation_grid


updateTime = ptime.time()
fps = 0
temp_index_y = 100
temp_index_x = 100


def update_data():
    global img, animation_grid, updateTime, fps, data1, ptr1, temp_index_y, temp_index_x
    data1 = np.roll(data1, -1)
    data = a.propagate(ecg=True)
    data = ani_convert(data)
    voltage = e.voltage(data, (temp_index_y, temp_index_x))
    data1[-1] = voltage
    ptr1 += 1
    curve.setData(data1)
    curve.setPos(ptr1, 0)
    time.sleep(1/120.)  # gives larger more stable fps.
    img.setImage(data.T, levels=(0, 50))

    QtCore.QTimer.singleShot(1, update_data)
    now = ptime.time()
    fps2 = 1.0 / (now - updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1

    # print "%0.1f fps" % fps

update_data()

vb = view.vb


def mouse_moved(evt):
    global temp_index_y, temp_index_x
    pos = evt[0]  # using signal proxy turns original arguments into a tuple
    if view.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        index_x = int(mousePoint.x())
        index_y = int(mousePoint.y())
        if index_x >= 0 and index_x <= 200 and index_y >= 0 and index_y <= 200:
            label.setText("y: %s, x: %s" % (index_y, index_x))
        temp_index_y = index_y
        temp_index_x = index_x

proxy = pg.SignalProxy(view.scene().sigMouseMoved, rateLimit=60, slot=mouse_moved)

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
