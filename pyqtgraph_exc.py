import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import basic_propagate as bc
import time
import analysis_theano as at

print '[ani, exc_num]'

choice = raw_input("Choose output: ")

if choice == 'ani':
    a = bc.Heart(0.14)
    a.set_pulse(220)


    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget()
    win.show()
    win.setWindowTitle('animation')
    view = win.addViewBox()
    view.setAspectLocked(True)

    view.setAspectLocked(True)
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
        if data == []:            # could use <if not individual_data.any():> but this is more readable.
            return animation_grid
        else:
            indices = np.unravel_index(data, a.shape)
            for ind in range(len(indices[0])):
                animation_grid[indices[0][ind]][indices[1][ind]] = 50
            return animation_grid

    updateTime = ptime.time()
    fps = 0

    def updateData():
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

        QtCore.QTimer.singleShot(1, updateData)
        now = ptime.time()
        fps2 = 1.0 / (now - updateTime)
        updateTime = now
        fps = fps * 0.9 + fps2 * 0.1

        print "%0.1f fps" % fps

    updateData()

    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

if choice == 'exc_num':

    a = bc.Heart(0.10)
    a.set_pulse(220)
    win = pg.GraphicsWindow()
    win.setWindowTitle('Number of excited cells')
    p1 = win.addPlot()
    data1 = np.zeros(1000)
    curve = p1.plot(data1, pen=pg.mkPen('w', width=2))
    ptr1 = 0
    win.nextRow()
    p2 = win.addPlot()
    p2.setDownsampling(mode='peak')
    p2.setClipToView(True)
    p2.setRange(xRange=[-100, 0])
    data2 = np.empty(1000)
    curve2 = p2.plot(pen=pg.mkPen('w', width=2))


    def update1():
        global data1, data2, ptr1
        data1 = np.roll(data1, -1)
        frame = a.propagate(real_time=True)
        data1[-1] = frame
        data2[ptr1] = frame
        ptr1 += 1
        curve.setData(data1)
        curve.setPos(ptr1, 0)

        if ptr1 >= data2.shape[0]:
            tmp = data2
            data2 = np.empty(data2.shape[0] * 2)
            data2[:tmp.shape[0]] = tmp
        curve2.setData(data2[:ptr1])
        curve2.setPos(-ptr1, 0)


    def update():
        update1()


    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1)

    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

# if choice == "ecg":
#
#         a = bc.Heart(0.10)
#         a.set_pulse(220)
#
#         win = pg.GraphicsWindow()
#         win.setWindowTitle('ECG')
#         p1 = win.addPlot()
#         p1.setDownsampling(mode='peak')
#         p1.setClipToView(True)
#         p1.setRange(xRange=[-100, 0])
#         data1 = np.empty(100)
#         curve1 = p1.plot(pen=pg.mkPen('w', width=2))
#         ptr1 = 0
#
#
#         def ecg_update():
#             global data1, ptr1
#
#             frame = ecg_data(convert(a.propagate(ecg=True)), 1)
#             data1[ptr1] = frame
#             ptr1 += 1
#
#             if ptr1 >= data1.shape[0]:
#                 tmp = data1
#                 data1 = np.empty(data1.shape[0] * 2)
#                 data1[:tmp.shape[0]] = tmp
#             curve1.setData(data1[:ptr1])
#             curve1.setPos(-ptr1, 0)
#
#
#         def update():
#             ecg_update()
#
#
#         def convert(data):
#             """
#             Converts all the file data into arrays that can be animated.
#             :return:
#             """
#             animation_grid = np.zeros(a.shape)
#             if data == []:  # could use <if not individual_data.any():> but this is more readable.
#                 return np.reshape(np.array(animation_grid), (1, 200, 200))
#             else:
#                 indices = np.unravel_index(data, a.shape)
#                 for i in range(len(indices[0])):
#                     animation_grid[indices[0][i]][indices[1][i]] = 50
#                 return np.reshape(np.array(animation_grid), (1, 200, 200))
#
#
#         def course_grain(excitation_grid, cg_factor):
#             """ excitation_grid should be list of 2d arrays in time order where each 2d array
#             is the animation state of the system at time t. The excitation_grid
#             of a system can be obtained  using b = animator.Visual('file_name'), selecting your
#             desired animation range and then exporting excitation_grid = b.animation_data.
#
#             cg_factor is the unitless factor corresponding to the number of small original cells
#             along each side of the new course grained cell.
#             e.g. If a 200x200 array is processed with cg_factor = 5, the new course grained array
#             will be shape 40x40 where each new cell corresponds to the net excitations from 5x5
#             sections of the original array."""
#
#             exc = np.array(excitation_grid).astype('float')  # Asserts data type of imported excitation_grid
#             filt = np.ones((cg_factor, cg_factor), dtype='float')  # Square matrix of ones in shape of course_grained cells.
#             norm = cg_factor ** 2  # Number of original cells in each course grained cell
#             a = T.dtensor3('a')  # Theano requires us to specify data types. dtensor3 is a 3d tensor of float64's
#             b = T.dmatrix('b')  # Matrix of float64's
#             z = conv2d(a, b, subsample=(cg_factor, cg_factor)) / norm  # This specifies the function to process.
#             #               Convolution with subsample step length results in course grained matrices
#             f = function([a, b], z)  # Theano function definition where inputs ([a,b]) and outputs (z) are specified
#             return f(exc, filt)  # Returns function with excitation_grid and filter as output
#
#
#         def ecg_data(excitation_grid, cg_factor, probe_pos=None):  # By default probe at (shape[0]/2,shape[1]/2)
#             """Returns ECG time series from excitation grid which is list of system state matrix at
#             each time step. This can either come from b = animator.Visual('file_name') -> b.animation_data,
#             or can be course grained using 'course_grain' If data has been course grained, this must be
#             specified in cg_factor to ensure distance between cells are correctly adjusted. Probe position
#             can be specified as a tuple of course grained coordinates ints (y,x). If probe_pos == None, probe
#             will be placed in centre of tissue. """
#
#             shape = np.shape(excitation_grid)
#             exc = excitation_grid.astype('float')
#             ex = T.dtensor3('ex')  # Theano variable definition
#             z1 = 50 - ex  # Converts excitation state to time state counter.
#             # i.e. excited state = 0, refractory state 40 -> 50 - 40 = 10
#             # State voltage conversion with theano
#             z2 = (((((50 - z1) ** 0.3) * T.exp(-(z1 ** 4) / 1500000) + T.exp(-z1)) / 4.2) * 110) - 20
#             f = function([ex], z2)
#             exc = f(exc) * (cg_factor ** 2)
#
#             if probe_pos != None:
#                 # If y coordinate of probe is not in tissue centre,
#                 # this will roll matrix rows until probe y coordinate is in central row
#                 exc = np.roll(exc, (shape[1] / 2) - probe_pos[0], axis=1)
#
#             x_dif = np.gradient(exc, axis=2)
#             y_dif = np.gradient(exc, axis=1)
#             x_dist = np.zeros_like(x_dif[0])
#             y_dist = np.zeros_like(y_dif[0])
#
#             for i in range(len(x_dist[0])):
#                 x_dist[:, i] = i
#             for i in range(len(y_dist)):
#                 y_dist[i] = i
#             if probe_pos == None:
#                 x_dist -= (shape[2] / 2)
#             else:
#                 x_dist -= probe_pos[1]
#             y_dist -= (shape[1] / 2)
#             net_x = x_dist * x_dif
#             net_y = y_dist * y_dif
#             net = net_x + net_y
#             z = 3
#             den = (((cg_factor * x_dist) ** 2) + ((cg_factor * y_dist) ** 2) + (z ** 2)) ** 1.5
#             for i in range(len(net)):
#                 try:
#                     return np.sum(net[i] / den)
#                 except:
#                     pass
#
#
#         timer = pg.QtCore.QTimer()
#         timer.timeout.connect(update)
#         timer.start(50)
#
#         if __name__ == '__main__':
#             import sys
#
#             if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#                 QtGui.QApplication.instance().exec_()