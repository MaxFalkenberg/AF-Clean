import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import basic_propagate as bc

for outputs in ['base_ani', 'exc_num']:
    print outputs

choice = raw_input("Choose output: ")

if choice == 'exc_num':

    a = bc.Heart(0.15)
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
