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
