import numpy as np
import propagate

def af_starts(start,end):
    nu = []
    af_initiated = []
    af_start_time = []
    for i in np.arange(start,end,0.005):
        temp1 = []
        temp2 = []
        nu.append(i)
        print 'nu = ' , i
        for j in range(100):
            a = propagate.Heart(nu = i, delta = 0.05,eps = 0.05,rp = 50,count_excited = True)
            a.set_pulse(220)
            x,y = a.propagate(1000000)
            temp1.append(x)
            temp2.append(y)
        af_initiated.append(temp1)
        af_start_time.append(temp2)
    return nu,af_initiated,af_start_time

def ecg_data(excitation_grid):
    voltage_grid = []
    twenty = np.ones((200,200))* 20
    rp = np.ones((200,200)) * 50
    for i in excitation_grid:
        voltage_grid.append(twenty - (110./50) * (rp - i))
    x_dif = []
    y_dif = []
    x_zero = np.zeros((200,1))
    y_zero = np.zeros((1,200))
    for i in voltage_grid:
        x_dif.append(np.append(np.diff(i),x_zero,axis = 1))
        y_dif.append(np.append(np.diff(i,axis = 0),y_zero, axis = 0))
    print np.shape(x_dif[0]),np.shape(y_dif[0])
    x_dist = np.zeros((200,200))
    y_dist = np.zeros((200,200))

    for i in range(len(x_dist)):
        x_dist[:,i] = i
        y_dist[i] = i
    x_dist -= 100
    y_dist -= 100
    net = []
    den = ((x_dist ** 2) + (y_dist ** 2)) ** 1.5
    den += 1
    for i in range(len(x_dif)):
        net.append(x_dist * x_dif[i] + y_dist * y_dif[i])
    ecg_values = []
    for i in range(len(net)):
        try:
            ecg_values.append(np.sum(net[i]/den))
        except:
            pass

    return ecg_values#, den, net
