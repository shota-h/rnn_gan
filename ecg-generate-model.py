import numpy as np
import matplotlib.pyplot as plt

def main():
    length = 10000
    fs = 100
    a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
    b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
    # theta = np.array([-np.pi/3, -np.pi/12, 0, np.pi/12, np.pi/2])
    theta = np.array([-np.pi/3, -np.pi/12, np.pi/24, np.pi/12, np.pi/2])
    dt = 1/fs
    # w = 2*np.pi*dt*0.5
    w = 2*np.pi/80
    f2 = 1/2
    A = 0.00015
    x = np.zeros([length, 1])
    y = np.zeros([length, 1])
    y[0] = -1
    z = np.zeros([length, 1])
    for t in range(length-1):
        x[t+1] = (1+dt*(1-np.sqrt(x[t]**2 + y[t]**2)))*x[t] - dt*w*y[t]
        y[t+1] = (1+dt*(1-np.sqrt(x[t]**2 + y[t]**2)))*y[t] + dt*w*x[t]
        z0 = A*np.sin(2*np.pi*f2*(t+1)*dt)
        Theta = np.arctan2(y[t], x[t])
        for i in range(5):
            dtheta = (Theta - theta[i]) % (2*np.pi)
            z[t+1] += a[i]*dtheta*np.exp(-(dtheta**2)/(2*b[i]**2))
        z[t+1] = dt*(z[t+1] - (z[t] - z0)) + z[t]
    plt.plot(x,y,'.')
    plt.show()
    plt.plot(z[:])
    plt.show()


if __name__ == '__main__':
    main()
