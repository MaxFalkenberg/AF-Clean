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

print data['Machine Learning Models']

ECG_start = data['ECG Start']  # (x, y)
ECG_end = data['ECG End']  # (x, y), y loop - ('NA', 'Y LOOP'), X loop - ('X Loop', y)
rotor_position = data['Rotor Position']  # (x, y)
ECG_count = data['ECG Counter']  # Total
Constraint_check = data['Constraint Check']  # either 0, 1, 3
Zero_check = data['Zero Check']  # either 0, 1, 3

zipped_data = zip(ECG_start, ECG_end, rotor_position, ECG_count, Constraint_check, Zero_check)

yLoop_number = ECG_end.count(('NA', 'Y LOOP'))
xLoop_number = [x[0] for x in ECG_end].count('X LOOP')

succesful_locations = [x for x in zipped_data if type(x[1][0]) is not str]
unsuccesful_locations = [x for x in zipped_data if type(x[1][0]) is str]

# Number of ECG statistics
total_ecg_average = np.mean([x[3] for x in succesful_locations])
total_ecg_std = np.std([x[3] for x in succesful_locations], ddof=1)


# Distance between rotor and final ecg position
def distance(rotor, ecg):
    """
    :param rotor: tuple
    :param ecg: tuple
    :return:
    """
    x_vector, y_vector = vector_distance(rotor, ecg)
    return np.sqrt(y_vector**2 + x_vector**2)


def vector_distance(rotor, ecg):
    """
    :param rotor:
    :param ecg:
    :return:
    """
    x_vector = ecg[0] - rotor[0]
    y_vector = ecg[1] - rotor[1]
    if y_vector > 100:
        y_vector -= 200
    if y_vector <= -100:
        y_vector += 200

    return x_vector, y_vector


distances = [distance(x[2], x[1]) for x in succesful_locations]
end_vector_dis = [vector_distance(x[2], x[1]) for x in succesful_locations]
end_abs_x_dis = [np.abs(x[0]) for x in end_vector_dis]
end_abs_y_dis = [np.abs(x[1]) for x in end_vector_dis]
x_limits = np.arange(-4,33)
y_limits = np.arange(-3,4)
target_true = []
target_dist = []
for i in end_vector_dis:
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


start_vector_dis = [vector_distance(x[2], x[0]) for x in zipped_data if type(x[1]) is tuple]
start_x_dis = [x[0] for x in start_vector_dis]
start_y_dis = [x[1] for x in start_vector_dis]

pred_prob =float(len(succesful_locations))/float(len(ECG_start))
tp = np.mean(target_true)
true_pred = tp * pred_prob
fail_mean = np.mean(target_dist[target_true != 1])
fail_std = np.std(target_dist[target_true != 1])
fail_median = np.median(np.sort(target_dist[target_true != 1]))


print '{:.2f}% of rotors found.'.format(len(succesful_locations)/float(len(ECG_start)) * 100)
print '{:.2f}% Y Loops.'.format(yLoop_number/float(len(ECG_start)) * 100)
print '{:.2f}% X Loops.'.format(xLoop_number/float(len(ECG_start)) * 100)


print "Average total number of ECGS to find rotor: {:.3f} ".format(total_ecg_average) \
      + u"\u00B1" + " {:.3f}".format(total_ecg_std)
print "True Positives when predicting: {:.2f}% ".format(tp * 100)
print "Successful Predictions: {:.2f}% ".format(true_pred * 100)
print "Mean fail distance from rotor: {:.2f} ".format(fail_mean) \
      + u"\u00B1" + " {:.2f}".format(fail_std)
print "Median fail distance from rotor: {:.2f}".format(fail_median)


plt.figure(figsize=(10, 5))
n = plt.hist(distances, bins=100)
plt.vlines(np.mean(distances), ymin=0, ymax=max(n[0]), colors='r', linestyles='dashed', linewidths=2)
plt.title('Rotor Distance Histogram.')
plt.xlabel('Distance from final probe to rotor')
plt.ylabel('Counts')
#
# fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# yloop_xvec = [(x[2][0] - x[0][0]) for x in zipped_data if x[1] == "Y LOOP"]
# yloop_yvec = [(x[2][1] - x[0][1]) for x in zipped_data if x[1] == "Y LOOP"]
# bx1.set_title('Y loops')
# bx1.set_ylabel('Counts')
# bx1.set_xlabel('Starting x vector from rotor')
# bx1.hist(yloop_xvec, bins=50)
# bx2.set_xlabel('Starting y vector from rotor')
# bx2.hist(yloop_yvec, bins=50)
#
# fig3, (cx1, cx2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# xloop_xvec = [(x[2][0] - x[0][0]) for x in zipped_data if x[1] == "X LOOP"]
# xloop_yvec = [(x[2][1] - x[0][1]) for x in zipped_data if x[1] == "X LOOP"]
# cx1.set_title('X loops')
# cx1.set_ylabel('Counts')
# cx1.set_xlabel('Starting x vector from rotor')
# cx1.hist(xloop_xvec, bins=50)
# cx2.set_xlabel('Starting y vector from rotor')
# cx2.hist(xloop_yvec, bins=50)
#
plt.figure()
count_grid = np.zeros((200, 200))
for x in end_vector_dis:
    count_grid[99 + x[1]][99 + x[0]] += 1
plt.imshow(count_grid, interpolation='nearest', origin="lower")
#
# plt.figure()
# succ_yecg_num = [x[3][0] for x in zipped_data if type(x[1]) is tuple]
# plt.scatter(start_x_dis, succ_yecg_num)
plt.show()
