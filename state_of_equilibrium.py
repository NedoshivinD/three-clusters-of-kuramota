import numpy as np
from scipy.optimize import root
import joblib 
import os

Nstream = 8
os.chdir("three-clusters-of-kuramota/res")

def func(p, M, K, alpha):
    f = np.zeros([2])
    x = p[0]
    y = p[1]
    f[0] = (M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha)
    f[1] = (N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-(M)*np.sin(y-x-alpha)
    return f
def res_sist(M, K, alpha): 
    all_sol = []   
    x = np.linspace(0, 2*np.pi, 10)
    y = np.linspace(0, 2*np.pi, 10)
    for i in x:
        for j in y:
            sol = root(func, [i, j], args = (M, K, alpha), method = 'hybr')
            xar, yar = sol.x
            xar = round(xar, 5)
            yar = round(yar, 5)
            if(([xar, yar, M, K, alpha] not in all_sol) and (abs(xar) < 2*np.pi) and (abs(yar) < 2*np.pi)):
                all_sol.append([xar, yar, M, K, alpha])
                

    return all_sol

N = 3
Marray = np.linspace(1, N-1, N-1, dtype = 'int')
Karray = np.linspace(N-1, 1, N-1, dtype = 'int')
alarray = np.linspace(0, np.pi-0.000001, N)
res = joblib.Parallel(n_jobs = Nstream) (joblib.delayed(res_sist) (M, K, alpha) for M in Marray for K in Karray for alpha in alarray if (M+K<N))
name = "res_n_3.txt"
print(len(res))
with open(name,"w",encoding="utf-8") as file:
    for x in res:
        for i in x:
            file.write(str(i) + "\n")

# print(func([2.72015, 2.1834], 8, 5, 6.283185307179586))
# [-58.53381, -31.41593, 8, 0, 5.585053606381854] [14.38588, 12.56637, 8, 0, 3.9269908169872414] 2.72015, 2.1834, 8, 5, 6.283185307179586