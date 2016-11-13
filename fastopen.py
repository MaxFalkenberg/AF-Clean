import numpy as np
import h5py
from numpy.fft import rfft
from numpy.fft import irfft
import matplotlib.pyplot as plt
import animator as an
import propagate as pr

testfile = h5py.File('SingleSource_Dataset1_ITT100_P60.h5','r')
group = testfile.get('Index: 902')
cp = np.array(group['Crit Position'])
probes = np.array(group['Probe Positions'])
ecg_vals = np.array(group['ECG'])
ecg_0 = ecg_vals[0]
fft_0 = rfft(ecg_0)

def sample(number = 0):
    ecg = ecg_vals[number]
    plt.figure()
    plt.plot(ecg)
    ft = rfft(ecg)
    ft_max10 = np.argsort(np.absolute(ft))[-5:]
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ft.size, d = 1.)
    freq_main = np.fft.rfftfreq(ft.size, d = 1.)[ft_max]
    period = int(1. / freq_main)
    ft[ft_max + 1:] = 0
    ift = irfft(ft)
    start = np.argmax(ift[:(2*period) - 1])
    end = start + (2 * period)
    sample_ecg = ecg[start:end]

    plt.figure()
    plt.plot(ecg)
    plt.plot(ift)
    plt.figure()
    plt.plot(sample_ecg)
    plt.show()
    return sample_ecg

def fft_sample(number = 0):
    sam = sample(number = number)
    ft = rfft(sam)
    plot_fft(number = number)
    plt.figure()
    x = np.fft.rfftfreq(sam.size, d = 1.)
    plt.stem(x, np.abs(ft)/np.sum(np.abs(ft)))
    plt.xlabel('Frequency, Hz', fontsize = 18)
    plt.ylabel('Amplitude', fontsize = 18)
    plt.show()

def plot_ecg(number = 0,save = False):
    global ecg_0
    plt.figure()
    plt.plot(ecg_vals[number])
    plt.xlabel('Time, t', fontsize = 18)
    plt.ylabel('Voltate, mV', fontsize = 18)
    plt.figure()
    plt.plot(np.gradient(ecg_vals[number]))
    plt.xlabel('Time, t', fontsize = 18)
    plt.ylabel('ECG Voltage Gradient')
    if save != False:
        plt.savefig(str(filename.pdf))
    plt.show()

def plot_fft(number = 0,save = False):
    global ecg_0
    f = rfft(ecg_vals[number])
    x = np.fft.rfftfreq(ecg_vals[number].size, d = 1.)
    plt.stem(x,np.abs(f)/np.sum(np.abs(f)))
    plt.xlabel('Frequency, Hz', fontsize = 18)
    plt.ylabel('Amplitude', fontsize = 18)
    if save != False:
        plt.savefig(str(filename.pdf))
    plt.show()

def plot_both(number = 0):
    plt.figure()
    plot_ecg(number = number)
    plt.figure()
    plot_fft(number = number)

def reverse_animate():
    global cp
    y = cp/200
    x = cp - (y * 200)
    print y,x
    a = pr.Heart(nu = 1, delta = 0., rp = 50)
    a.set_pulse(60,[[y],[x]])
    a.propagate(400)
    a.save('ANIMATE_TEMPORARY')
    b = an.Visual('ANIMATE_TEMPORARY', mode = 'Auto')
    b.show_animation()
    plt.show()

def remove_highf(number, filter = None):
    a = np.fft.rfft(ecg_vals[number])
    ab = np.absolute(a)
    plt.figure()
    plt.plot(a)
    if filter != None:
        a[filter:] = 0.
    plt.plot(a)
    plt.xlabel('Frequency, Hz', fontsize = 18)
    plt.ylabel('Amplitude', fontsize = 18)
    plt.figure()
    plt.plot(ecg_vals[number])
    plt.xlabel('Time, t', fontsize = 18)
    plt.ylabel('Voltage', fontsize = 18)
    if filter != None:
        b = np.fft.irfft(a)
        plt.figure()
        plt.xlabel('Time, t', fontsize = 18)
        plt.ylabel('Voltage', fontsize = 18)
        plt.plot(b)
    plt.show()
