import matplotlib.pyplot as plt
import numpy as np
import os, sys
from scipy import signal, interpolate


filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/segment-ecg'.format(filedir)
loadpath = '{0}/ecg-generate-model'.format(filedir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


def resample(x, A):
    N1 = np.linspace(0, 1, x.shape[0])
    N2 = np.linspace(0, 1, A)
    f = interpolate.interp1d(N1, x)
    return f(N2)



def main():
    ecg = np.load('{0}/generate-ecg.npy'.format(loadpath))
    peaks = np.load('{0}/ecg-peak.npy'.format(loadpath))
    R_peak = np.where(peaks == 3)[0]
    ecg1 = ecg[R_peak[1]-5:R_peak[2]-5]
    ecg2 = resample(ecg1, 96)
    plt.plot(ecg2)
    plt.show()


if __name__ == '__main__':
    main()
