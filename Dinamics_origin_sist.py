from ast import arg
import numpy as np
from scipy import integrate
import joblib 
import matplotlib.pyplot as plt
from lib import *
from Dinamics import read_f


t = np.linspace(0, 100, 100)

def func(p, t, M, K, alpha):
    f = np.zeros([6])
    u = p[0]
    o = p[1]
    k = p[2]
    fi1 = p[3]
    fi2 = p[4]
    fi3 = p[5]

    f[0] = u #fi1 с точкой
    f[1] = o #fi2 с точкой
    f[2] = k #fi3 с точкой
    f[3] = 1/m * (omega + 1/N * (- K * np.sin(alpha) + M * np.sin(fi2 - fi1 - alpha) + (N - K - M) * np.sin(fi3 - fi1 - alpha)) - u)
    f[4] = 1/m * (omega + 1/N * (- M * np.sin(alpha) + K * np.sin(fi1 - fi2 - alpha) + (N - M - K) * np.sin(fi3 - fi2 - alpha)) - o)
    f[5] = 1/m * (omega + 1/N * (- (N - M - K) * np.sin(alpha) + K * np.sin(fi1 - fi3 - alpha) + M * np.sin(fi2 - fi3 - alpha)) - k)
    return f

def func_res(fi2, fi3, M, K, alpha):
    fi2 = fi2 + eps
    fi3 = fi3 + eps
    u0 = 0
    o0 = 0
    k0 = 0
    star_point = [fi1, fi2, fi3, u0, o0, k0]
    res = integrate.odeint(func, star_point, t, args=(M, K, alpha))
    print(res)
    return(res)

def PlotOnPlane(t, point):
    res = func_res(point[0], point[1], point[2], point[3], point[4])  #x, y, K, M, alpha
    plt.plot(t, res[:, 0], label="fi1")
    plt.plot(t, res[:, 1], label="fi2", linestyle = '--')
    plt.plot(t, res[:, 2], label="fi3", linestyle = '-.')
    plt.xlim(0, 100)
    plt.ylim(-10, 20)
    plt.legend()
    plt.show() 

fi1 = 1
point = (0.0, 0.0, 1.0, 7.0, np.pi/3)
x = point[0]
y = point[1]
fi2 = fi1 - x
fi3 = fi1 - y

PlotOnPlane(t, point)