from ast import arg
import numpy as np
from scipy import integrate
import joblib 
import matplotlib.pyplot as plt
from lib import *



def syst(p, t, M, K, alpha):
    f = np.zeros([4])
    x,y,z,w = p
    f[0] = 1/(N*m)*((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha) - z)
    f[1] = 1/(N*m)*((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-(M)*np.sin(y-x-alpha) - z)
    f[2] = z
    f[3] = w
    return f

def integr(x,y,M, K, alpha):
    x0 = x + eps
    y0 = y + eps
    z0 = 0
    w0 = 0
    star_point = [x0, y0, z0, w0]
    res = integrate.odeint(syst, star_point, t, args=(M, K, alpha))
    res = res.reshape(len(res),4)
    return(res)

def read_f(name_f):
    with open(name_f,"r") as file:
        arr = file.read()
        print(arr)


if __name__ == "__main__":
    Marray = np.linspace(0, N-1, N, dtype = 'int')
    Karray = np.linspace(N-1, 0, N, dtype = 'int')
    alarray = np.linspace(0, 2*np.pi, N)

    res = []
    t = np.linspace(0, 100, 100)

    res = integr(6.28319, 6.28319, 8.0, 0.0, 2.356194490192345)  
    plt.plot(t, res[:, 0], label="x")
    plt.plot(t, res[:, 1], label="y", linestyle = '--')
    plt.xlim(0, 100)
    plt.ylim(-10, 20)
    plt.legend()
    plt.show()



# ress = []
# with open("E:\\work\\three_klasters_kuramoto\\three-clusters-of-kuramota\\res\\res_n_12.txt") as file:
#     for line in file:
#         ress.append(lib.razb_str(line.rstrip()))
#         # print(line.rstrip())
        




