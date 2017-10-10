import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import sys

H = 60.0
ALPHA = np.sqrt(H/60.0)
a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])*ALPHA
# theta = np.array([-np.pi/3.0, -np.pi/12.0, 0.0, np.pi/12.0, np.pi/2.0])*ALPHA
theta = np.array([-np.pi/3*np.sqrt(ALPHA), -np.pi/12.0*ALPHA, 0.0, np.pi/12.0*ALPHA, np.pi/2*np.sqrt(ALPHA)])
A = 0.005
f2 = 0.25
length = 10000
fs = 512
dt = 1.0/fs



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


def dxdt(x, y, w):
    alpha = 1-np.sqrt(x**2+y**2)
    return alpha*x - w*y


def dydt(x, y, w):
    alpha = 1-np.sqrt(x**2+y**2)
    return alpha*y + w*x


def dzdt(x, y, z, zbase):
    Theta = np.arctan2(y, x)
    dtheta = (Theta - theta) % (2.0*np.pi)
    # C = -np.sum(a*dtheta*np.exp(-0.5*(dtheta/b)**2))
    return -1.0*np.sum(a*dtheta*np.exp(-0.5*(dtheta/b)**2)) - (z-zbase)


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
    Hw0 = [Hw[:n/2],Hw[n/2:0:-1]]
    Sw = fs_rr/2*np.sqrt(Hw0)

    ph0 = 2*np.pi*np.random.uniform(low=0,high=1,size=(n/2-1))
    ph = [0, ph0, 0, ph0[-1:0:-1]]
    SwC = Sw * exp(i*ph)
    x = (1/n)*(np.fft.ifft(SwC)).real
    x_std = np.std(x)
    ratio = rr_std/x_std
    return rr_mean + x*ratio


def runge(x0, y0, z0):
    rr = rrprocess(1,2,1,1,1,60,1,1,256)
    plt.plot(rr)
    plt.show()
    sys.exit()
    w = 2*np.pi/1.0
    x = x0
    y = y0
    z = z0
    x1 = x0
    y1 = y0
    z1 = z0
    X = [x]
    Y = [y]
    Z = [z]
    X1 = [x]
    Y1 = [y]
    Z1 = [z]
    z0 = A*np.sin(2*np.pi*f2*np.arange(length)*dt)
    for t in np.arange(length-1):
        d1 = dfdt(x1, y1, z1, z0[t+1], w)
        d2 = dfdt(x1+d1[0]*dt*0.5, y1+d1[1]*dt*0.5, z1+d1[2]*dt*0.5, z0[t+1], w)
        d3 = dfdt(x1+d2[0]*dt*0.5, y1+d2[1]*dt*0.5, z1+d2[2]*dt*0.5, z0[t+1], w)
        d4 = dfdt(x1+d3[0]*dt, y1+d3[1]*dt, z1+d3[2]*dt, z0[t+1], w)
        if t == 0:
            dfdt(x1, y1, z1, z0[t+1], w, i=0)
            # print((d1[0] + 2*d2[0] + 2*d3[0] + d4[0])*(dt/6.0))
            # print((d1[1] + 2*d2[1] + 2*d3[1] + d4[1])*(dt/6.0))
            # print((d1[2] + 2*d2[2] + 2*d3[2] + d4[2])*(dt/6.0))
        x1 += (d1[0] + 2*d2[0] + 2*d3[0] + d4[0])*(dt/6.0)
        y1 += (d1[1] + 2*d2[1] + 2*d3[1] + d4[1])*(dt/6.0)
        z1 += (d1[2] + 2*d2[2] + 2*d3[2] + d4[2])*(dt/6.0)
        X1.append(x1)
        Y1.append(y1)
        Z1.append(z1)
        # dx1 = dxdt(x, y, w)
        # dx2 = dxdt(x+dx1*dt*0.5, y+dx1*dt*0.5, w)
        # dx3 = dxdt(x+dx2*dt*0.5, y+dx2*dt*0.5, w)
        # dx4 = dxdt(x+dx3*dt, y+dx3*dt, w)
        # dy1 = dydt(x, y, w)
        # dy2 = dydt(x+dy1*dt*0.5, y+dy1*dt*0.5, w)
        # dy3 = dydt(x+dy2*dt*0.5, y+dy2*dt*0.5, w)
        # dy4 = dydt(x+dy3*dt, y+dy3*dt, w)
        # x += (dx1 + 2*dx2 + 2*dx3 + dx4)*(dt/6.0)
        # y += (dy1 + 2*dy2 + 2*dy3 + dy4)*(dt/6.0)
        #
        # d1 = dzdt(x, y, z, z0[t+1])
        # d2 = dzdt(x+d1*dt*0.5, y+d1*dt*0.5, z+d1*dt*0.5, z0[t+1])
        # d3 = dzdt(x+d2*dt*0.5, y+d2*dt*0.5, z+d2*dt*0.5, z0[t+1])
        # d4 = dzdt(x+d3*dt, y+d3*dt, z+d3*dt, z0[t+1])
        # z += (d1 + 2*d2 + 2*d3 + d4)*(dt/6.0)
        # X.append(x)
        # Y.append(y)
        # Z.append(z)
    # Z = Z[0:-1:2]
    # Z = 1.6*(Z - min(Z))/(max(Z) - min(Z))-0.4
    # plt.plot(Z,'-')
    plt.plot(Z1,'-')
    plt.show()


def main(x0, y0, z0):
    x = np.zeros([length, 1])
    y = np.zeros([length, 1])
    z = np.zeros([length, 1])
    x[0] = x0
    y[0] = y0
    z[0] = z0
    z0 = A*np.sin(2*np.pi*f2*np.arange(length)*dt)
    for t in range(length-1):
        x[t+1] = (1+dt*(1-np.sqrt(x[t]**2 + y[t]**2)))*x[t] - dt*w*y[t]
        y[t+1] = (1+dt*(1-np.sqrt(x[t]**2 + y[t]**2)))*y[t] + dt*w*x[t]
        Theta = np.arctan2(y[t], x[t])
        for i in range(5):
            dtheta = (Theta - theta[i]) % (2*np.pi)
            z[t+1] += a[i]*dtheta*np.exp(-(dtheta**2)/(2*b[i]**2))
        z[t+1] = dt*(z[t+1] - (z[t] - z0[t])) + z[t]
    plt.plot(z[:],'.-')
    # plt.show()


if __name__ == '__main__':
    # runge(1.0, 0.0, -0.04)
    runge(1.0, 0.0, 0.04)
    # v0 = [1.0,0,0.04]
    # t = np.arange(length)*dt
    # v = odeint(func, v0, t, args=(a, b, w, theta))
    # main(0,-1,0)
