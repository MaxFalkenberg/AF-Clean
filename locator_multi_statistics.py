"""
Works out statistics for rotor data
"""

import sys
import cPickle
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv

rotor_datafile = args[1]
with open('%s.p' % rotor_datafile, 'rb') as f:
    data = cPickle.load(f)

# print data['Machine Learning Models']

ECG_start = data['ECG Start']  # (x, y)
ECG_end = data['ECG End']  # (x, y), y loop - ('NA', 'Y LOOP'), X loop - ('X Loop', y)
rotor_position = data['Rotor Position']  # (x, y)
ECG_count = data['ECG Counter']  # Total
Y_constraint_check = data['Y Constraint Check']  # either 0, 1, 3
X_constraint_check = data['X Constraint Check']  # either 0, 1, 3
Y_zero_check = data['Y Zero Check']
X_zero_check = data['X Zero Check']
Y_loop_check = data['Y Loop']
X_loop_check = data['X Loop']

zipped_data = zip(ECG_start, ECG_end, rotor_position, ECG_count, Y_constraint_check, X_constraint_check,
                  Y_zero_check, X_zero_check, Y_loop_check, X_loop_check)

# ECG statistics
average_ecg_1 = np.mean([x[3][0] for x in zipped_data])
std_ecg_1 = np.std([x[3][0] for x in zipped_data], ddof=1)
average_ecg_2 = np.mean([x[3][1] for x in zipped_data])
std_ecg_2 = np.std([x[3][1] for x in zipped_data], ddof=1)

pure_locators = [x for x in zipped_data if x[4] == [0, 0] and x[5] == [0, 0] and x[6] == [0, 0] and x[7] == [0, 0] and x[8] == [0, 0] and x[9] == [0, 0]]
no_constraint_limit_locators = [x for x in zipped_data if x[4] == [0, 0] and x[5] == [0, 0]]
no_loop_locators = [x for x in zipped_data if x[8] == [0, 0] and x[9] == [0, 0]]
no_zero_locators = [x for x in zipped_data if x[6] == [0, 0] and x[7] == [0, 0]]


# Distance between rotor and final ecg position
def distance(vectors):
    """
    :param rotors: tuple
    :param ecgs: tuple
    :return:
    """
    x_vector1 = vectors[0][0]
    x_vector2 = vectors[1][0]
    y_vector1 = vectors[0][1]
    y_vector2 = vectors[1][1]

    distance_1 = np.sqrt(y_vector1**2 + x_vector1**2)
    distance_2 = np.sqrt(y_vector2**2 + x_vector2**2)
    return distance_1, distance_2


def y_vector_wrapper(y_vector):
    """
    :return:
    """
    if y_vector > 100:
        y_vector -= 200
    if y_vector <= -100:
        y_vector += 200
    return y_vector


def vector_distance(rotor1, ecg1, rotor2, ecg2):
    """
    :param rotor1:
    :param ecg1:
    :param rotor2:
    :param ecg2:
    :return:
    """
    y_vector1 = y_vector_wrapper(ecg1[1] - rotor1[1])  # Link A
    y_vector2 = y_vector_wrapper(ecg2[1] - rotor1[1])  # Link A
    y_vector3 = y_vector_wrapper(ecg1[1] - rotor2[1])  # Link B
    y_vector4 = y_vector_wrapper(ecg2[1] - rotor2[1])  # Link B

    if np.abs(y_vector1) < np.abs(y_vector2):
        x_vector1 = ecg1[0] - rotor1[0]
        true_y_vector1 = y_vector1
    else:
        x_vector1 = ecg2[0] - rotor1[0]
        true_y_vector1 = y_vector2

    if np.abs(y_vector3) < np.abs(y_vector4):
        x_vector2 = ecg1[0] - rotor2[0]
        true_y_vector2 = y_vector3
    else:
        x_vector2 = ecg2[0] - rotor2[0]
        true_y_vector2 = y_vector4

    distance_1 = (x_vector1, true_y_vector1)
    distance_2 = (x_vector2, true_y_vector2)

    return distance_1, distance_2

vectors = [vector_distance(x[2][0], x[1][0], x[2][1], x[1][1]) for x in zipped_data]
vector_unp = [x for sublist in vectors for x in sublist]
pdistances = [distance(v) for v in vectors]
distances = [x for sublist in pdistances for x in sublist]  # Distances from the ref point.

average_distance = np.mean(distances)
std_distances = np.std(distances, ddof=1)

x_limits = np.arange(-4,33)
y_limits = np.arange(-3,4)
target_true = []
target_dist = []
for i in vector_unp:
    x = i[0] in x_limits
    y = i[1] in y_limits
    t = x * y
    target_true.append(t)
    x_dif = np.min(np.absolute(i[0] - x_limits))
    y_dif = np.min(np.absolute(i[1] - y_limits))
    d = ((x_dif ** 2) + (y_dif ** 2)) ** 0.5
    target_dist.append(d)
target_true = np.array(target_true)
target_dist = np.array(target_dist)

tp = np.mean(target_true)
fail_mean = np.mean(target_dist[target_true != 1])
fail_std = np.std(target_dist[target_true != 1])
fail_median = np.median(np.sort(target_dist[target_true != 1]))
#
#
# print '{:.2f}% of rotors found.'.format(len(succesful_locations)/float(len(ECG_start)) * 100)
# print '{:.2f}% Y Loops.'.format(yLoop_number/float(len(ECG_start)) * 100)
# print '{:.2f}% X Loops.'.format(xLoop_number/float(len(ECG_start)) * 100)
#
#
print "Average total number of ECGS to find first rotor: {:.3f} ".format(average_ecg_1) \
      + u"\u00B1" + " {:.3f}".format(std_ecg_1)
print "Average total number of ECGS to find second rotor: {:.3f} ".format(average_ecg_2) \
      + u"\u00B1" + " {:.3f}".format(std_ecg_2)
print "True Positives when predicting: {:.2f}% ".format(tp * 100)
print "Mean fail distance from rotor: {:.2f} ".format(fail_mean) \
      + u"\u00B1" + " {:.2f}".format(fail_std)
print "Median fail distance from rotor: {:.2f}".format(fail_median)
#
#
# plt.figure(figsize=(10, 5))
# n = plt.hist(distances, bins=100)
# plt.vlines(np.mean(distances), ymin=0, ymax=max(n[0]), colors='r', linestyles='dashed', linewidths=2)
# plt.title('Rotor Distance Histogram.')
# plt.xlabel('Distance from final probe to rotor')
# plt.ylabel('Counts')
# #
# # fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# # yloop_xvec = [(x[2][0] - x[0][0]) for x in zipped_data if x[1] == "Y LOOP"]
# # yloop_yvec = [(x[2][1] - x[0][1]) for x in zipped_data if x[1] == "Y LOOP"]
# # bx1.set_title('Y loops')
# # bx1.set_ylabel('Counts')
# # bx1.set_xlabel('Starting x vector from rotor')
# # bx1.hist(yloop_xvec, bins=50)
# # bx2.set_xlabel('Starting y vector from rotor')
# # bx2.hist(yloop_yvec, bins=50)
# #
# # fig3, (cx1, cx2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# # xloop_xvec = [(x[2][0] - x[0][0]) for x in zipped_data if x[1] == "X LOOP"]
# # xloop_yvec = [(x[2][1] - x[0][1]) for x in zipped_data if x[1] == "X LOOP"]
# # cx1.set_title('X loops')
# # cx1.set_ylabel('Counts')
# # cx1.set_xlabel('Starting x vector from rotor')
# # cx1.hist(xloop_xvec, bins=50)
# # cx2.set_xlabel('Starting y vector from rotor')
# # cx2.hist(xloop_yvec, bins=50)
# #
plt.figure()
count_grid = np.zeros((300, 300))
for x in vector_unp:
    count_grid[149 + x[1]][149 + x[0]] += 1
plt.imshow(count_grid, interpolation='nearest', origin="lower")
# #
# # plt.figure()
# # succ_yecg_num = [x[3][0] for x in zipped_data if type(x[1]) is tuple]
# # plt.scatter(start_x_dis, succ_yecg_num)
plt.show()
