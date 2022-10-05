import numpy as np
from scipy.optimize import root
import joblib 

# N = 9
Nstream = 8

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
            if(([xar, yar, M, K, alpha] not in all_sol) and (abs(xar) <= round(2*np.pi, 5)) and (abs(yar) <= round(2*np.pi, 5))):
                all_sol.append([xar, yar, M, K, alpha])
                

    return all_sol
for n in range(3,18,3):
    N = n
    Marray = np.linspace(0, N-1, N, dtype = 'int')
    Karray = np.linspace(N-1, 0, N, dtype = 'int')
    alarray = np.linspace(0, 2*np.pi, N)
    res = joblib.Parallel(n_jobs = Nstream) (joblib.delayed(res_sist) (M, K, alpha) for M in Marray for K in Karray for alpha in alarray)
    name = f"res_n_{n}.txt"
    # name1 = "bed_res.txt"
    with open(name,"w",encoding="utf-8") as file:
        for x in res:
            for i in x:
                # if (int(i[0]%round(np.pi, 5)) != 0 and int(i[1]%round(np.pi, 5)) != 0):
                file.write(str(i) + "\n")

# print(func([2.72015, 2.1834], 8, 5, 6.283185307179586))
# [-58.53381, -31.41593, 8, 0, 5.585053606381854] [14.38588, 12.56637, 8, 0, 3.9269908169872414] 2.72015, 2.1834, 8, 5, 6.283185307179586