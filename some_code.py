from re import L
from lib import *
import numpy as np
from numpy import linalg as LA
from Dinamics import *

def matrix(x, y, K, M, alpha):
    f = []
    f.append([0, 0, 1, 0])
    f.append([0, 0, 0, 1])
    f.append([1/(N*m)*(-M*np.cos(x+alpha) - K*np.cos(x - alpha) - (N - M - K)*np.cos(x-y-alpha)),
            1/(N*m)*(-(N-M-K)*np.cos(y + alpha) + (N-M-K)*np.cos(x-y-alpha)), -1/m, 0])
    f.append([1/(N*m)*(-M*np.cos(x+y) + M*np.cos(y-x-alpha)),
            1/(N*m)*(-M*np.cos(x+y)-K*np.cos(y-alpha)-(N-M-K)*np.cos(y+alpha)-M*np.cos(y-x-alpha)),
            0, -1/m])
    return(f)

def yakobi(x, y, K, M, alpha):    
    matrix_yak = matrix(x, y, K, M, alpha)
    lam, vect = LA.eig(matrix_yak)
    return(lam)    

for i in yakobi(0.0, 0.0, 1.0, 1.0, 0.0):
    print(i)

def sinfaza():
    os.chdir("res")
    ress = []
    with open("stability_sost_ravn.txt") as file:
        for line in file:
            ress.append(razb_str(line.rstrip()))

    sinfaz = []
    for str in ress:
        for j in range(2):
            if str[j] == 0 and str[j+1] == 0:
                sinfaz.append(str)
    with open("Sinfaz.txt","w",encoding="utf-8") as file:
        for x in range(len(sinfaz)):
            print(sinfaz[x])
            # file.write(str(x) + "\n")
    os.chdir("ust_graphs")
    for i in range(50):
        res = func_res(sinfaz[i][0], sinfaz[i][1], sinfaz[i][2], sinfaz[i][3], sinfaz[i][4])  
        plt.plot(t, res[:, 0], label="x")
        plt.plot(t, res[:, 1], label="y", linestyle = '--')
        plt.xlim(0, 100)
        plt.ylim(-10, 20)
        plt.legend()
        plt.savefig(f"graph{i+1}")
        plt.close()
# sinfaza()