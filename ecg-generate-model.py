import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy import signal, interpolate
import sys, os

H = 60.0
ALPHA = np.sqrt(H/60.0)
a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
# a = np.array([1.2, -5.0, 30.0, -7.5, 2])
b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])*ALPHA
theta = np.array([-np.pi/3*np.sqrt(ALPHA), -np.pi/12.0*ALPHA, 0.0, np.pi/12.0*ALPHA, np.pi/2*np.sqrt(ALPHA)])
A = 0.005
f2 = 0.25
length = 10000*4
# length = 1000
# fs = 512
fs = 96*2
# fs2 = 256
fs2 = 96
dt = 1.0/fs

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/ecg-generate-model'.format(filedir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


def func(v, t, a, b, w, theta):
    alpha = 1-np.sqrt(v[0]**2+v[1]**2)
    Theta = np.arctan2(v[1], v[0])
    dtheta = (Theta - theta) % (2.0*np.pi)
    zbase = A*np.sin(2*np.pi*f2*t)
    C = -np.sum(a*dtheta*np.exp(-0.5*(dtheta/b)**2))
    dx = alpha*v[0]-w*v[1]
    dy = alpha*v[1]+w*v[0]
    dz = C - (v[2]-zbase)
    return [dx, dy, dz]


def dfdt(x, y, z, zbase, w, i=1):
    alpha = 1-np.sqrt(x**2+y**2)
    Theta = np.arctan2(y, x)
    dtheta = (Theta - theta) % (2.0*np.pi)
    dtheta[dtheta > np.pi] = dtheta[dtheta > np.pi] - (2*np.pi)
    return [alpha*x - w*y, alpha*y + w*x, -1.0*np.sum(a*dtheta*np.exp(-0.5*(dtheta/b)**2)) - (z-zbase)]


def rrprocess(flo, fhi, flo_std, fhi_std, flfh_ratio, hr_mean, hr_std, fs_rr, n):
    w1 = 2*np.pi*flo
    w2 = 2*np.pi*fhi
    c1 = 2*np.pi*flo_std
    c2 = 2*np.pi*fhi_std
    sig1 = flfh_ratio
    sig2 = 1
    rrmean = 60/hr_mean
    rrstd = 60*hr_std/(hr_mean**2)

    df = fs_rr/n
    w = np.arange(length)*2*np.pi*df
    dw1 = w-w1
    dw2 = w-w2

    Hw1 = sig1*np.exp(-0.5*(dw1/c1)**2)/(np.sqrt(2*np.pi*c1**2))
    Hw2 = sig2*np.exp(-0.5*(dw2/c2)**2)/(np.sqrt(2*np.pi*c2**2))
    Hw = Hw1 + Hw2
    Hw0 = np.append(Hw[:int(n/2)],Hw[int(n/2)-1::-1],axis=0)
    Sw = fs_rr/2*np.sqrt(Hw0)

    ph0 = 2*np.pi*np.random.uniform(low=0,high=1,size=(int(n/2-1)))
    ph = np.zeros(ph0.shape[0]*2+2)
    ph[1:ph0.shape[0]+1] = ph0
    ph[ph0.shape[0]+2:] = -ph0[-1::-1]

    Sw = np.array(Sw)
    SwC = Sw * np.exp(-1j*ph)
    x = (1/n)*(np.fft.ifft(SwC)).real
    x_std = np.std(x)
    ratio = rrstd/x_std
    return rrmean + x*ratio


def detectpeaks(x, y, z, dtheta, fs_ecg):
    N = z.shape[0]
    ipeaks = np.zeros(N)
    theta = np.arctan2(y, x)
    ind0 = np.zeros(N)
    for i in range(N-1):
        a = (theta[i] <= dtheta) & (dtheta <= theta[i+1])
        j = np.where(a == True)
        if len(j[0]) != 0:
            d1 = dtheta[j[0]] - theta[i]
            d2 = theta[i+1] -dtheta[j[0]]
            if d1 < d2:
                ind0[i] = j[0] + 1
            else:
                ind0[i+1] = j[0] + 1
    d = np.ceil(fs_ecg/64)
    d = np.max([2,d])
    ind = np.zeros(N)
    zext = [np.min(z), np.max(z), np.min(z), np.max(z), np.min(z)]
    sext = [1, -1, 1, -1, 1]
    for i in range(5):
        ind1 = np.where(ind0 == (i+1))
        n = len(ind1[0])
        Z = np.ones([n,int(2*d+1)])*zext[i]*sext[i]
        for j in np.arange(-d,d):
            k = (0 <= ind1+j) & (ind1+j <= N-1)
            A = (ind1[0][k[0]]+j).astype('int32')
            Z[k[0],int(d+j)] = z[A]*sext[i]
        ivmax = np.argmax(Z, axis=1)
        iext = ind1 + ivmax-d
        ind[iext[0].astype('int32')] = i+1
    return ind


def make_datasets(s, peaks):
    nR = np.where(peaks == 3)
    nR = np.array(nR[0][1:])
    rand = np.random.randint(low=-5, high=5, size=(nR.shape[0]))
    seg_ecg = np.array([s[i+rand[j]-int(fs2/2):i+rand[j]+int(fs2/2)] for j, i in enumerate(nR)])
    if seg_ecg.shape[0] > 200:
        try:
            np.save('{0}/dataset/normal_dynamical_model.npy'.format(filedir), seg_ecg)
        except:
            print('not save it')
        else:
           print('save it')
        finally:
            pass


def main(x0 = 1.0, y0 = 0.0, z0 = 0.04):
    N = 256
    rr = rrprocess(0.1,0.25,0.01,0.01,1,60,1,1,N)
    ff = interpolate.interp1d(np.arange(N)/(N-1), rr)
    rr = ff(np.arange(N*fs)/(N*fs-1))
    w = 2*np.pi/(rr)
    x = x0
    y = y0
    z = z0
    X = [x]
    Y = [y]
    Z = [z]
    z0 = A*np.sin(2*np.pi*f2*np.arange(length)*dt)
    for t in np.arange(length-1):
        d1 = dfdt(x, y, z, z0[t+1], w[t])
        d2 = dfdt(x+d1[0]*dt*0.5, y+d1[1]*dt*0.5, z+d1[2]*dt*0.5, z0[t], w[t])
        d3 = dfdt(x+d2[0]*dt*0.5, y+d2[1]*dt*0.5, z+d2[2]*dt*0.5, z0[t], w[t])
        d4 = dfdt(x+d3[0]*dt, y+d3[1]*dt, z+d3[2]*dt, z0[t], w[t])
        x += (d1[0] + 2*d2[0] + 2*d3[0] + d4[0])*(dt/6.0)
        y += (d1[1] + 2*d2[1] + 2*d3[1] + d4[1])*(dt/6.0)
        z += (d1[2] + 2*d2[2] + 2*d3[2] + d4[2])*(dt/6.0)
        X.append(x)
        Y.append(y)
        Z.append(z)
    X = X[0:-1:2]
    Y = Y[0:-1:2]
    Z = Z[0:-1:2]
    Z = 1.6*(Z - min(Z))/(max(Z) - min(Z))-0.4
    peaks = detectpeaks(X, Y, Z, theta, fs2)
    # plt.plot(Z)
    # plt.show()
    # plt.figure()
    # ax = plt.subplot(111, projection='3d')
    # ax.plot(X[:400], Y[:400], Z[:400])
    # plt.savefig('{0}/trajectory.tif'.format(filepath))
    # plt.close()
    # plt.savefig('{0}/output.tif'.format(filepath))
    make_datasets(Z, peaks)

if __name__ == '__main__':
    main(1.0, 0.0, 0.04)
