import numpy as np
import matplotlib.pyplot as plt
import sys

H = 30
ALPHA = np.sqrt(H/60.0)
a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])*ALPHA
theta = np.array([-np.pi/3, -np.pi/12, 0, np.pi/12, np.pi/2])*ALPHA
f2 = 0.25
A = 0.1
length = 10000
fs = 256
dt = 1/fs
w = 2*np.pi*dt*0.05


def dxdt(x, y, w):
    alpha = 1-np.sqrt(x**2+y**2)
    return alpha*x-w*y

def dydt(x, y, w):
    alpha = 1-np.sqrt(x**2+y**2)
    return alpha*y+w*x


def dzdt(z, z0, Theta):
    dtheta = (Theta - theta) % (2*np.pi)
    C = np.sum(a*dtheta*np.exp(-dtheta**2/(2*b**2)))
    return C - (z-z0)


def runge(x0, y0, z0):
    X = []
    Y = []
    Z = []
    x = x0
    y = y0
    z = z0
    z0 = A*np.sin(2*np.pi*f2*np.arange(length)*dt)
    for t in range(length-1):
        d1 = dxdt(x, y, w)
        d2 = dxdt(x+d1*dt*0.5, y, w)
        d3 = dxdt(x+d2*dt*0.5, y, w)
        d4 = dxdt(x+d3*dt, y, w)
        x += (d1 + 2*d2 + 2*d3 + d4)*(dt/6.0)
        d1 = dydt(x, y, w)
        d2 = dydt(x, y+d1*dt*0.5, w)
        d3 = dydt(x, y+d2*dt*0.5, w)
        d4 = dydt(x, y+d3*dt, w)
        y += (d1 + 2*d2 + 2*d3 + d4)*(dt/6.0)
        Theta = np.arctan2(y, x)
        d1 = dzdt(z, z0[t], Theta)
        d2 = dzdt(z+d1*dt*0.5, z0[t], Theta)
        d3 = dzdt(z+d2*dt*0.5, z0[t], Theta)
        d4 = dzdt(z+d3*dt, z0[t], Theta)
        z += (d1 + 2*d2 + 2*d3 + d4)*(dt/6.0)
        X.append(x)
        Y.append(y)
        Z.append(z)
    # plt.plot(X,Y,'.')
    # plt.show()
    plt.plot(Z,'.-')
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
    # plt.plot(x,y,'.')
    # plt.show()
    plt.plot(z[:],'.-')
    # plt.show()


if __name__ == '__main__':
    # main(0,-1,0)
    runge(0,-1,0)
