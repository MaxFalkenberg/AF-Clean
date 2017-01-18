import propagate_singlesource as ps
import analysis_theano as at
import numpy as np
import matplotlib.pyplot as plt
from Functions import sampling_convert
# import pyqtgraph as pg
# import pyqtgraph.ptime as ptime
# from pyqtgraph.Qt import QtCore, QtGui
a = ps.Heart(fakedata=True)
a.set_pulse(220, [[100], [100]])
e = at.ECG(shape=(200, 200), probe_height=3)
raw_data = a.propagate(400)

converted_data = list()
grid = np.zeros(a.shape)
sampling_convert(raw_data, converted_data, shape=a.shape, rp=a.rp, animation_grid=grid)

ecg = e.solve(converted_data[100:])

probe_positions = e.probe_position

print probe_positions
print "Plotting..."
fig = plt.figure()
for index, i in enumerate(ecg):
    plt.plot(range(len(i)), i, linewidth=2.0, label='{0}'.format(probe_positions[index]))
# a = ps.Heart(fakedata=True)
# a.set_pulse(220, [[100], [100]])
# raw_data = a.propagate(1100)
#
# converted_data = list()
# grid = np.zeros(a.shape)
# convert(raw_data, converted_data)
#
# fig = plt.figure()
# for _ in range(5):
#     e = at.ECG(shape=(200, 200), probe_height=3)
#     ecg = e.solve(converted_data[100:])
#     plt.plot(range(len(ecg[0])), ecg[0], label='{0}'.format(e.probe_position))
#
plt.ylabel("Voltage")
plt.xlabel("Time Step")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.close()

# app = QtGui.QApplication([])
# win = pg.GraphicsLayoutWidget()
# win.show()
# win.setWindowTitle('Animation')
# view = win.addViewBox()
# view.setAspectLocked(True)
#
# view.setAspectLocked(True)
# img = pg.ImageItem(border='w')
# view.addItem(img)
#
# view.setRange(QtCore.QRectF(0, 0, 200, 200))
# animation_grid = np.zeros((200, 200))
#
# updateTime = ptime.time()
# fps = 0
# ptr = 0
#
#
# def update_data():
#     global img, animation_grid, ptr, updateTime, fps
#     ani_data = converted_data[ptr]
#     # time.sleep(1/120.)  # gives larger more stable fps.
#     img.setImage(ani_data.T, levels=(0, 50))
#     ptr += 1
#
#     if ptr == len(converted_data):
#         ptr = 0
#
#     QtCore.QTimer.singleShot(1, update_data)
#     now = ptime.time()
#     fps2 = 1.0 / (now - updateTime)
#     updateTime = now
#     fps = fps * 0.9 + fps2 * 0.1
#
#     # print "%0.1f fps" % fps
#
# update_data()
#
# if __name__ == '__main__':
#     import sys
#
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_
