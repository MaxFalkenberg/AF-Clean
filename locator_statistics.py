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
print '\n'

ECG_start = data['ECG Start']  # (x, y)
ECG_end = data['ECG End']  # (x, y), y loop - ('NA', 'Y LOOP'), X loop - ('X Loop', y)
rotor_position = data['Rotor Position']  # (x, y)
ECG_count = data['ECG Counter']  # Total
Constraint_check = data['Constraint Check']  # either 0, 1 - x axis, 2 - y axis,  3 - both
Zero_check = data['Zero Check']  # either 0, 1 - y jumps, 2 - x jumps,  3 - both

zipped_data = zip(ECG_start, ECG_end, rotor_position, ECG_count, Constraint_check, Zero_check)

yLoop_number = ECG_end.count(("NA", "Y LOOP"))
xLoop_number = [x[0] for x in ECG_end].count('X LOOP')
yconstraint_number = Constraint_check.count(1)
xconstraint_number = Constraint_check.count(2)
xyconstraint_number = Constraint_check.count(3)

succesful_locations = [x for x in zipped_data if type(x[1][0]) is not str]
unsuccesful_locations = [x for x in zipped_data if type(x[1][0]) is str]

positive_class_positions = [x for x in zipped_data if type(x[1][0]) is not str and x[4] == 0 and x[5] == 0]

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


def confusion(vector):
    """
    Determines if the values are true true positives or false positives.
    :param vector:
    :return:
    """
    result = [None, None]

    xabs = np.abs(vector[0])
    yabs = np.abs(vector[1])

    if xabs < 8:
        # TP
        result[0] = 1
    if xabs >= 8:
        # FP
        result[0] = 0
    if yabs < 3:
        # TP
        result[1] = 1
    if yabs >= 3:
        # FP
        result[1] = 0

    return result

distances = [distance(x[2], x[1]) for x in succesful_locations]
end_vector_dis = [vector_distance(x[2], x[1]) for x in succesful_locations]
end_abs_x_dis = [np.abs(x[0]) for x in end_vector_dis]
end_abs_y_dis = [np.abs(x[1]) for x in end_vector_dis]
start_vector_dis = [vector_distance(x[2], x[0]) for x in zipped_data if type(x[1]) is tuple]
start_x_dis = [x[0] for x in start_vector_dis]
start_y_dis = [x[1] for x in start_vector_dis]

# True positive/ True negative etc
vector_distances = [vector_distance(x[2], x[1]) for x in positive_class_positions]
xclass_confusion_data = [confusion(x)[0] for x in vector_distances]
xTP = xclass_confusion_data.count(1)
xFP = xclass_confusion_data.count(0)
yclass_confusion_data = [confusion(x)[1] for x in vector_distances]
yTP = yclass_confusion_data.count(1)
yFP = yclass_confusion_data.count(0)

print 'Total Number of rotors: %s' % len(rotor_position)
print '{:.2f}% of rotors had a predicited position.'.format(len(succesful_locations)/float(len(ECG_start)) * 100)
print '{:.2f}% resulted in Y Loops.'.format(yLoop_number/float(len(ECG_start)) * 100)
print '{:.2f}% resulted in X Loops.'.format(xLoop_number/float(len(ECG_start)) * 100)

print '\n'

print '{:.2f}% of rotors involved a full y constraint.'.format(yconstraint_number/float(len(Constraint_check)) * 100)
print '{:.2f}% of rotors involved a full x constraint.'.format(xconstraint_number/float(len(Constraint_check)) * 100)
print '{:.2f}% of rotors involved both a full x and full y constraint.'.format(xyconstraint_number/float(len(
    Constraint_check)) * 100)

print '\n'

print "Average number of ECG's to find each rotor: {:.3f} ".format(total_ecg_average) \
      + u"\u00B1" + " {:.3f}".format(total_ecg_std)
print "Mean distance from rotor: {:.3f} ".format(np.mean(distances)) \
      + u"\u00B1" + " {:.3f}".format(np.std(distances, ddof=1))
print "Mean y distance from rotor: {:.3f} ".format(np.mean(end_abs_y_dis)) \
      + u"\u00B1" + " {:.3f}".format(np.std(end_abs_y_dis, ddof=1))
print "Mean x distance from rotor: {:.3f} ".format(np.mean(end_abs_x_dis)) \
      + u"\u00B1" + " {:.3f}".format(np.std(end_abs_x_dis, ddof=1))

print '\n'

print '{:.2f}% y class true positive.'.format(yTP/float(len(positive_class_positions)) * 100)
print '{:.2f}% y class false positive.'.format(yFP/float(len(positive_class_positions)) * 100)
print '{:.2f}% x class true positive.'.format(xTP/float(len(positive_class_positions)) * 100)
print '{:.2f}% x class false positive.'.format(xFP/float(len(positive_class_positions)) * 100)

plt.figure(figsize=(10, 5))
n = plt.hist(distances, bins=30)
plt.vlines(np.mean(distances), ymin=0, ymax=max(n[0]), colors='r', linestyles='dashed', linewidths=2)
plt.title('Rotor Distance Histogram.')
plt.xlabel('Distance from final probe to rotor')
plt.ylabel('Counts')

plt.figure()
count_grid = np.zeros((200, 200))
for x in end_vector_dis:
    count_grid[99 + x[1]][99 + x[0]] += 1
plt.imshow(count_grid, interpolation='nearest', origin="lower")

plt.show()
plt.close()
