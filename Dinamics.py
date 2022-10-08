from ast import arg
import numpy as np
from scipy import integrate
import joblib 
import matplotlib.pyplot as plt
import lib

N = 9
Marray = np.linspace(0, N-1, N, dtype = 'int')
Karray = np.linspace(N-1, 0, N, dtype = 'int')
alarray = np.linspace(0, 2*np.pi, N)
m = 1
res = []
t = np.linspace(0, 1000, 10000)
def func(p, t, M, K, alpha):
    f = np.zeros([4])
    x = p[0]
    y = p[1]
    z = p[2]
    w = p[3]
    f[0] = 1/m*((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha) - z)
    f[1] = 1/m*((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-(M)*np.sin(y-x-alpha) - z)
    f[2] = z
    f[3] = w
    return f
def func_res(M, K, alpha):
    x0 = 2*np.pi
    y0 = 0
    z0 = 0
    w0 = 0
    star_point = [x0, y0, z0, w0]
    res = integrate.odeint(func, star_point, t, args=(M, K, alpha))
    res = res.reshape(len(res),4)
    return(res)
    # xar, yar = res
    # xar = round(xar, 5)
    # yar = round(yar, 5)
    # if(([xar, yar, M, K, alpha] not in all_sol) and (abs(xar) <= round(2*np.pi, 5)) and (abs(yar) <= round(2*np.pi, 5))):
    #     all_sol.append([xar, yar, M, K, alpha])
def read_f(name_f):
    with open(name_f,"r") as file:
        arr = file.read()
        print(arr)
# res = func_res(8, 0, 3.9269908169872414)   # [6.28319, -0.0, 8, 0, 3.9269908169872414]
# plt.plot(t, res[:,0])
# plt.show() 
# file = open("res_n_9.txt")
ress = []
with open("E:\\work\\three_klasters_kuramoto\\three-clusters-of-kuramota\\res\\res_n_12.txt") as file:
    for line in file:
        ress.append(lib.razb_str(line.rstrip()))
        # print(line.rstrip())

# print(ress[0])
for i in ress:
    print(i)



# print(razb_str(ress[0]))


