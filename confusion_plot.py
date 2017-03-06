""""
Plots the confusion matrix data from _conf_s.p files
keys in data dict are (y_scale, x_scale)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

datafile = raw_input("Confusion series data file to open: ")
with open("%s.p" % datafile, 'rb') as f:
    data = pickle.load(f)

# x and y axis list
y_coords = sorted(list(set([x[0] for x in data])))
y_ref = range(len(y_coords))
x_coords = sorted(list(set([x[1] for x in data])))
x_ref = range(len(x_coords))

recall_data = {k: i[1][1]/float((i[1][0] + i[1][1])) for (k, i) in data.items()}
plotting_data = np.zeros((len(x_coords), len(y_coords)))

for i in x_ref:
    line = plotting_data[i]
    for ii in y_ref:
        line[ii] = recall_data[(y_coords[ii], x_coords[i])]

fig = plt.figure()

for i in x_ref:
    plt.plot(y_coords, plotting_data[i], label="x scale: %s" % x_coords[i])

plt.title("Elliptical targets")
plt.xlabel("y scale")
plt.ylabel("Recall")
plt.legend(loc="upper left")
plt.show()
plt.close()

# print grid



# works out the recall data for each target shape.


# print recall_data
#
# for i in y_ref:
#     line = grid[i]
#     for ii in x_ref:
#         line[ii] = recall_data[(y_coords[i], x_coords[ii])]
#
# fig, ax = plt.subplots()
# ax.imshow(grid, interpolation="none", origin="lower")
# ax.set_xticks(x_coords)
# ax.set_yticks(y_coords)
# ax.set_aspect(2)
# plt.show()
# plt.close()