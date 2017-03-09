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
ECG_end = data['ECG End']  # (x, y)
rotor_position = data['Rotor Position']  # (x, y)
ECG_count = data['ECG Counter']  # Total
ECG_check = data['Check']

zipped_data = zip(ECG_start, ECG_end, rotor_position, ECG_count, ECG_check)
print zipped_data

yLoop_number = ECG_end.count('Y LOOP')
xLoop_number = [x[0] for x in ECG_end].count('X LOOP')
print yLoop_number
print xLoop_number

# yLoop_number = ECG_end.count('Y LOOP')
# xLoop_number = ECG_end.count('X LOOP')
#
# succesful_locations = [x for x in zipped_data if type(x[1]) is tuple]
# unsuccesful_locations = [x for x in zipped_data if type(x[1]) is not tuple]
#
# # Number of ECG statistics
# total_ecg_average = np.mean([sum(x[3]) for x in succesful_locations])
# total_ecg_std = np.std([sum(x[3]) for x in succesful_locations], ddof=1)
#
# # Distance between rotor and final ecg position
# def distance(rotor, ecg):
#     """
#     :param rotor: tuple
#     :param ecg: tuple
#     :return:
#     """
#     x_vector, y_vector = vector_distance(rotor, ecg)
#     return np.sqrt(y_vector**2 + x_vector**2)
#
#
# def vector_distance(rotor, ecg):
#     """
#     :param rotor:
#     :param ecg:
#     :return:
#     """
#     x_vector = ecg[0] - rotor[0]
#     y_vector = ecg[1] - rotor[1]
#     if y_vector > 100:
#         y_vector -= 200
#     if y_vector <= -100:
#         y_vector += 200
#
#     return x_vector, y_vector
#
#
# distances = [distance(x[2], x[1]) for x in zipped_data if type(x[1]) is tuple]
# end_vector_dis = [vector_distance(x[2], x[1]) for x in zipped_data if type(x[1]) is tuple]
# end_abs_x_dis = [np.abs(x[0]) for x in end_vector_dis]
# end_abs_y_dis = [np.abs(x[1]) for x in end_vector_dis]
# start_vector_dis = [vector_distance(x[2], x[0]) for x in zipped_data if type(x[1]) is tuple]
# start_x_dis = [x[0] for x in start_vector_dis]
# start_y_dis = [x[1] for x in start_vector_dis]
#
# # distance_filterd = [x for x in distances if x<100]
# # test = [x for x in distances if x>100]
# # print len(distance_filterd)
# # print np.mean(distance_filterd)
# # print np.std(distance_filterd)
#
# print '{:.2f}% of rotors found.'.format(len(succesful_locations)/float(len(ECG_start)) * 100)
# print '{:.2f}% Y Loops.'.format(yLoop_number/float(len(ECG_start)) * 100)
# print '{:.2f}% X Loops.'.format(xLoop_number/float(len(ECG_start)) * 100)
#
# print "Average total number of ECGS to find rotor: {:.4f} ".format(total_ecg_average) \
#       + u"\u00B1" + " {:.4f}".format(total_ecg_std)
# print "Average number of ECGS to find X-Axis: {:.4f} ".format(y_ecg_average) \
#       + u"\u00B1" + " {:.4f}".format(y_ecg_std)
# print "Average number of ECGS to find Y-Axis: {:.4f} ".format(x_ecg_average) \
#       + u"\u00B1" + " {:.4f}".format(x_ecg_std)
# print "Mean distance from rotor: {:.4f} ".format(np.mean(distances)) \
#       + u"\u00B1" + " {:.4f}".format(np.std(distances, ddof=1))
# print "Mean y distance from rotor: {:.4f} ".format(np.mean(end_abs_y_dis)) \
#       + u"\u00B1" + " {:.4f}".format(np.std(end_abs_y_dis, ddof=1))
# print "Mean x distance from rotor: {:.4f} ".format(np.mean(end_abs_x_dis)) \
#       + u"\u00B1" + " {:.4f}".format(np.std(end_abs_x_dis, ddof=1))

# plt.figure(figsize=(10, 5))
# n = plt.hist(distances, bins=100)
# plt.vlines(np.mean(distances), ymin=0, ymax=max(n[0]), colors='r', linestyles='dashed', linewidths=2)
# plt.title('Rotor Distance Histogram.')
# plt.xlabel('Distance from final probe to rotor')
# plt.ylabel('Counts')
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
# plt.figure()
# count_grid = np.zeros((400, 400))
# for x in end_vector_dis:
#     count_grid[199 + x[1]][199 + x[0]] += 1
# plt.imshow(count_grid, interpolation='nearest', origin="lower")
#
# plt.figure()
# succ_yecg_num = [x[3][0] for x in zipped_data if type(x[1]) is tuple]
# plt.scatter(start_x_dis, succ_yecg_num)
# plt.show()

