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

   
ress = []
with open("C:\\Users\\nedos\\Desktop\\scientific work\\three-clusters-of-kuramota\\res\\res_n_9.txt") as file:
    for line in file:
        ress.append(razb_str(line.rstrip()))


lam = yakobi(4.71239, 4.71239, 8.0, 0.0, 5.497787143782138)
print(lam)
# with open("C:\\Users\\nedos\\Desktop\\scientific work\\three-clusters-of-kuramota\\res\\stability_sost_ravn.txt","w") as file:
#     for i in range(len(ress)):
#         lam = yakobi(ress[i][0], ress[i][1], ress[i][2], ress[i][3], ress[i][4]) 
#         g = 1
#         for j in lam:
#             if j.real < 0:
#                 continue
#             else:
#                 g = 0
#                 break  
#         if g == 1:
#             file.write(str(ress[i]) + '\n')

# with open("C:\\Users\\nedos\\Desktop\\scientific work\\three-clusters-of-kuramota\\res\\non_stability_sost_ravn.txt","w") as file:
#     for i in range(len(ress)):
#         lam = yakobi(ress[i][0], ress[i][1], ress[i][2], ress[i][3], ress[i][4]) 
#         g = 1
#         for j in lam:
#             if j.real < 0:
#                 continue
#             else:
#                 g = 0
#                 break  
#         if g != 1:
#             file.write(str(ress[i]) + '\n')



# with open("C:\\Users\\nedos\\Desktop\\scientific work\\three-clusters-of-kuramota\\ust\\res_n_9.txt","w") as file:
#     for v in ress:
#         res = func_res(v[0],v[1],v[2],v[3],v[4])
#         if abs(res[0][0] - res[-1][0]) < 0.1:
#             file.write(str(v) + "  -  Ust\n")
#         else:
#             file.write(str(v) + "  -  Ne ust\n")