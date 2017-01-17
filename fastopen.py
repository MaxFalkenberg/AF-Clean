import numpy as np
import h5py
from numpy.fft import rfft
from numpy.fft import irfft
import matplotlib.pyplot as plt
import animator as an
import propagate as pr

inp = str(raw_input('H5 file to open:'))
testfile = h5py.File(inp, 'r')
group = testfile.get('Index: 0')
cp = np.array(group['Crit Position'])
probes = np.array(group['Probe Positions'])
ecg_vals = np.array(group['ECG'])
ecg_0 = ecg_vals[0]
fft_0 = rfft(ecg_0)
y,x = np.unravel_index(cp,(200,200))

def cp_vector(y_probe,x_probe):
    x_vector = int(x_probe) - x
    y_vector = int(y_probe) - y
    if y_vector > 100:
        y_vector -= 200
    elif y_vector <= -100:
        y_vector += 200

    r = ((x_vector ** 2) + (y_vector ** 2)) ** 0.5
    c = (x_vector + (1j * y_vector)) /r
    theta = np.angle(c)
    return r,c,theta

y, x = np.unravel_index(cp, (200, 200))
pythag = np.zeros((200, 200), dtype='float')
x_grid = np.copy(pythag)
y_grid = np.copy(pythag)
y_mid = float(len(y_grid) / 2)
for i in range(len(pythag)):
    x_grid[:, i] = i
    y_grid[i] = i
x_grid -= float(x)
y_grid -= y_mid
pythag += ((x_grid ** 2) + (y_grid ** 2)) ** 0.5
dist_grid = np.roll(pythag, int(y_mid + y), axis=0)


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
    plt.figure()
    plt.plot(np.gradient(sample_ecg))
    plt.show()
    return sample_ecg


def fft_sample(number=0):
    sam = sample(number = number)
    ft = rfft(sam)
    plot_fft(number = number)
    plt.figure()
    x = np.fft.rfftfreq(sam.size, d = 1.)
    plt.stem(x, np.abs(ft)/np.sum(np.abs(ft)))
    plt.xlabel('Frequency, Hz', fontsize = 18)
    plt.ylabel('Amplitude', fontsize = 18)
    plt.show()


def plot_ecg(number=0, save=False):
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


def plot_fft(number=0, save=False):
    global ecg_0
    f = rfft(ecg_vals[number])
    x = np.fft.rfftfreq(ecg_vals[number].size, d = 1.)
    plt.stem(x,np.abs(f)/np.sum(np.abs(f)))
    plt.xlabel('Frequency, Hz', fontsize = 18)
    plt.ylabel('Amplitude', fontsize = 18)
    if save != False:
        plt.savefig(str(filename.pdf))
    plt.show()


def plot_both(number=0):
    plt.figure()
    plot_ecg(number=number)
    plt.figure()
    plot_fft(number=number)


def reverse_animate():
    global cp
    y = cp/200
    x = cp - (y * 200)
    print y, x
    a = pr.Heart(nu=1, delta=0., rp=50)
    a.set_pulse(60, [[y], [x]])
    a.propagate(400)
    a.save('ANIMATE_TEMPORARY')
    b = an.Visual('ANIMATE_TEMPORARY', mode='Auto')
    b.show_animation()
    plt.show()


def remove_highf(number, filter=None):
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


def feature_extract(number):
    """Extracts features for the current itteration's ECG at the probe position
    corresponding to probes[number]. Not currently written to return values in a
    particular format."""
    ecg = ecg_vals[number]
    crit_point = cp #Index of critical point
    dist = dist_grid[int(probes[number][0])][int(probes[number][1])] #Distance of probe from CP

    ft = rfft(ecg)  # Real valued FT of original ECG
    ft_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_max10 = np.argsort(ft_abs)[-10:]  # Finds 10 largest frequency fundamentals
    ft_max = np.min(ft_max10)
    freq = np.fft.rfftfreq(ft.size, d=1.)
    freq_main = np.fft.rfftfreq(ft.size, d=1.)[ft_max]
    # FEATURE (Should be the same for all ECGs if correctly sampled.)
    period = int(1. / freq_main)
    ft2 = np.copy(ft)
    ft2[ft_max + 1:] = 0
    ift = irfft(ft2)
    start = np.argmax(ift[:(2*period) - 1])
    end = start + (2 * period)
    sample_ = ecg[start:end]  # Crops original ECG according to fundamental frequency.

    ft_samp = rfft(sample_)  # Real valued FT of sample ECG
    freq_samp = np.fft.rfftfreq(ft.size, d=1.)
    ft_samp_abs = np.absolute(ft)  # Takes absolute value of FT
    ft_samp_max10 = np.argsort(ft_abs)[-10:]  # Finds 10 largest frequency fundamentals

    grad = np.gradient(sample_)

    # FEATURE: Maximum value of sample ECG
    max_value = np.max(sample_)
    # FEATURE: Minimum value of sample ECG
    min_value = np.min(sample_)
    # FEATURE: Difference of the above
    minmax_dif = max_value - min_value
    # FEATURE: Sample ECG intensity defined as sum of absolute voltages
    sample_int = np.sum(np.absolute(sample_))
    # FEATURE (Should be the same for all ECGs. If this is differnt from usual sample is wrong.)
    sample_len = len(sample_)

    # FEATURE: Maximum of first order gradient of ECG
    grad_max = np.max(grad)
    # FEATURE: Minimum of first order gradient of ECG
    grad_min = np.min(grad)
    # FEATURE: Difference of the above
    grad_diff = grad_max - grad_min
    # FEATURE: Argument at gradient Minimum
    grad_argmin = np.argmin(grad)
    # FEATURE: Argument at gradient Maximum
    grad_argmax = np.argmax(grad)
    # FEATURE: Difference in Max and Min arguments. Gives idea of ECG curvature.
    grad_argdiff = grad_argmax - grad_argmin

    # FEATURE: Largest 10 frequencies in sample ECG. Largest first.
    largest_ft_freq = freq_samp[ft_samp_max10[::-1]]
    # FEATURE: Absolute values of largest 10 freqs
    largest_ft_mag = ft_samp_abs[ft_samp_max10[::-1]]
    # FEATURE: Sum of absolute values
    largest_sum = np.sum(ft_samp_abs[ft_samp_max10[::-1]])
    # FEATURE: Absolute values normalised by sum.
    largest_ft_rel_mag = largest_ft_mag / largest_sum

    print max_value
    print min_value
    print minmax_dif
    print sample_int
    print sample_len
    print grad_max
    print grad_min
    print grad_diff
    print grad_argmin
    print grad_argmax
    print grad_argdiff
    print largest_ft_freq
    print largest_ft_mag
    print largest_sum
    print largest_ft_rel_mag
    print dist

feature_extract(6)
