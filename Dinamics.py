from ast import arg
import numpy as np
from scipy import integrate
import joblib 
import matplotlib.pyplot as plt
from lib import *
import os


os.chdir("three-clusters-of-kuramota")

Marray = np.linspace(0, N-1, N, dtype = 'int')
Karray = np.linspace(N-1, 0, N, dtype = 'int')
alarray = np.linspace(0, np.pi, N)

res = []
t = np.linspace(0, 100, 100)

def func(p, t, M, K, alpha):
    f = np.zeros([4])
    x = p[0]
    y = p[1]
    z = p[2]
    w = p[3]
    f[0] = 1/m*(1/N * ((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha)) - z)
    f[1] = 1/m*(1/N * ((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-(M)*np.sin(y-x-alpha)) - w)
    f[2] = z
    f[3] = w
    return f
    
def func_res(x,y,M, K, alpha):
    x0 = x + eps
    y0 = y + eps
    z0 = 0
    w0 = 0
    star_point = [x0, y0, z0, w0]
    res = integrate.odeint(func, star_point, t, args=(M, K, alpha))
    res = res.reshape(len(res),4)
    return(res)

def read_f(name_f):
    with open(name_f,"r") as file:
        arr = file.read()
        print(arr)
        
def PlotOnPlane(t, point):
    res = func_res(point[0], point[1], point[2], point[3], point[4])  #x, y, K, M, alpha
    plt.plot(t, res[:, 0], label="x")
    plt.plot(t, res[:, 1], label="y", linestyle = '--')
    plt.xlim(0, 100)
    plt.ylim(-10, 20)
    plt.legend()
    plt.show() 

def plotandwright(t):
    os.chdir("res")
    ress = []
    with open("res_n_3.txt") as file: #stability_sost_ravn.txt
        for line in file:
            ress.append(razb_str(line.rstrip()))
    os.chdir("ust_graphs")
    for i in range(21):
        res = func_res(ress[i][0], ress[i][1], ress[i][2], ress[i][3], ress[i][4])  
        plt.plot(t, res[:, 0], label="x")
        plt.plot(t, res[:, 1], label="y", linestyle = '--')
        plt.xlim(0, 100)
        plt.ylim(-10, 20)
        plt.legend()
        plt.savefig(f"graph{i+1}")
        plt.close()

point = (4.18879, 2.0944, 1.0, 1.0, 1.5707958267948965)
PlotOnPlane(t, point)

# plotandwright(t)



# ress = []
# with open("E:\\work\\three_klasters_kuramoto\\three-clusters-of-kuramota\\res\\res_n_12.txt") as file:
#     for line in file:
#         ress.append(lib.razb_str(line.rstrip()))
#         # print(line.rstrip())
        



# [1.5708, 1.5708, 8, 0, 3.9269908169872414] /
# [1.81951, 0.0, 8, 0, 3.9269908169872414] /
# 0.0, 0.0, 8, 0, 3.9269908169872414 - та точка
# альфа д.б. до пи