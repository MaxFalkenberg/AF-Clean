import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Point import Point
import numpy as np
import basic_propagate as bc
import time
import analysis_theano as at

nu_value = float(raw_input('Choose a Nu value: '))
a = bc.Heart(nu_value)
a.set_pulse(220)

app = QtGui.QApplication([])
win = pg.GraphicsWindow()
win.show()
win.setWindowTitle('animation')
view = win.addPlot()
# view.setAspectLocked(True)
img = pg.ImageItem(border='w')
view.addItem(img)
view.setRange(QtCore.QRectF(0, 0, 200, 200))

animation_grid = np.zeros((200, 200))

win.nextRow()
p1 = win.addPlot()
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


def update_data():
    global img, animation_grid, updateTime, fps, data1, ptr1
    data1 = np.roll(data1, -1)
    data, len_data = a.propagate(both=True)
    data = ani_convert(data)
    data1[-1] = len_data
    ptr1 += 1
    curve.setData(data1)
    curve.setPos(ptr1, 0)
    # time.sleep(1/120.)  # gives larger more stable fps.
    img.setImage(data.T, levels=(0, 50))

    QtCore.QTimer.singleShot(1, update_data)
    now = ptime.time()
    fps2 = 1.0 / (now - updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1

    # print "%0.1f fps" % fps

vLine = pg.InfiniteLine(angle=90, movable=False)
hLine = pg.InfiniteLine(angle=0, movable=False)
view.addItem(vLine, ignoreBounds=True)
view.addItem(hLine, ignoreBounds=True)

vb = view.vb

def mouseMoved(evt):
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    if view.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        index_x = int(mousePoint.x())
        index_y = int(mousePoint.y())
        if index_x >= 0 and index_y >= 0:
            print (index_x, index_y)

proxy = pg.SignalProxy(view.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

update_data()

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()