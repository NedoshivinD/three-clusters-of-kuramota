import numpy as np
import threading
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from supp_func import *
import random



n_job = 4

def iter_din(t, start_point , par):
    
    alpha,beta,k1,k2,w = par
    phi = np.zeros(len(start_point)//2)
    v = np.zeros(len(start_point)//2)
    
    for i in range(len(start_point)//2):
        phi[i] = start_point[i]
        v[i] = start_point[i+len(phi)]

    s = 0
    
    lens = len(phi)
    f = np.zeros(len(phi)*2)
    
    for j in range(len(phi)):
        for phi_i in phi:
            s += k1*np.sin(phi_i - phi[j]  - alpha) + k2*np.sin(2*(phi_i - phi[j])  - beta)
        
        f[j] = v[j]
        f[j+len(phi)]=s/lens + w - v[j]

        s = 0
    return f

def din_thr_map(phi,v,par,t,t_max):
    start_point = np.zeros(len(phi)*2)
    for i in range(len(phi)):
        start_point[i] = phi[i]
        start_point[i+len(phi)] = v[i]

    res = solve_ivp(iter_din,[0,t_max],start_point, args=[par],rtol= 10e-10,atol=10e-10) # t_eval=t,
    
    return res.y

def up_arr(start_phi,arr,N,num_elems,eps):
    res = np.array([])
    tmp = np.zeros(num_elems//N)
    
    if N>num_elems:
        num_elems = N
    
    
    
    razb = [arr[1],N-arr[1]]
    
    
    if np.pi - np.abs(arr[0])<1e-5 and np.pi - np.abs(arr[0]) > -1e-5:
        arr[0] = np.pi * np.sign(arr[0])
    

    if np.pi - np.abs(arr[2])<1e-5:
        arr[2] = np.pi * np.sign(arr[2])
    
    
    if np.pi - np.abs(arr[3])<1e-5:
        arr[3] = np.pi * np.sign(arr[3])
    
    
    tmp +=start_phi
    
    for i in range(razb[0]):
        res = np.append(res,tmp+random.uniform(-1, 1)*1e-1)
    
    tmp-= start_phi
    
    tmp = tmp+arr[0]
    
    for i in range(razb[1]):
        res = np.append(res, start_phi - tmp + random.uniform(-1, 1)*1e-1)
        
    tmp = tmp-arr[0]
    
    return res
    
def work(param):
    
    phi,eps,alpha,beta,k1,k2,t_max = param
    
    v = np.zeros(len(phi))
    v = v + 1e-1
    
    w = 1
    t = np.linspace(0,t_max,t_max)
    a = din_thr_map(phi,v,[alpha,beta,k1,k2,w],t,t_max)
    
    matrix = np.array([])
    for i in range(len(phi)):
        matrix = np.append(matrix,a[i])
    
    matrix = matrix.reshape((len(phi),len(matrix)//len(phi)))

    l = len(phi) - 1
    while l>=0:
        # if l != 19:
        matrix[l] = matrix[l] - matrix[0]
        l-=1

    
    # matrix[5:10] = 1e-10
    # matrix[12] = 1e10
    matrix+=eps
    
    matrix = np.angle(np.exp(1j*matrix))
    print(matrix)
    # plt.yticks(yticks,size=7)
    # plt.figure(figsize=(cm_to_inch(40),cm_to_inch(60)))
    # plt.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*100,0,len(phi)*10], aspect=4)
    fig, ax = plt.subplots()
    # matrix = np.angle(np.exp(1j*matrix))
    # print(matrix)
    p = ax.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect='auto')
    fig.colorbar(p,label=r'$\varphi_1 - \varphi_i$')
    plt.xlabel("t")
    plt.ylabel("N")
    plt.show()
    
    
    # plt.savefig('tmp.svg')
    
def heat_map(par):
    tmp = razb_config()
    eps = 1e-5
    low_arr = par
    start_phi = 1
    eps = 1e-7
    phi1 = up_arr(start_phi,low_arr,tmp[0],tmp[0],eps)
    alpha = low_arr[2]
    beta = low_arr[3]
    k1 = low_arr[4]
    k2 = low_arr[5]
    
    t_amx = 10000
    
    work([phi1,eps,alpha,beta,k1,k2,t_amx])
    
if __name__ == "__main__":
    

    tmp = razb_config()
    eps = 1e-5

    low_arr = [1.609267, 2, 0.0, 2.0944, 1, 1]
    
    start_phi = 1
    eps = 1e-1
    phi1 = up_arr(start_phi,low_arr,tmp[0],tmp[0],eps)
    alpha = low_arr[2]
    beta = low_arr[3]
    k1 = low_arr[4]
    k2 = low_arr[5]
    t_amx = 10000

    work([phi1,eps,alpha,beta,k1,k2,t_amx])

    # param = [1.959579, 2, 2.0944, 3.14159, 1, 1]
    # way= f"2_garmonic\\res\\n_{tmp[0]}\\tmp\\{param}\\"
    # arr = [1,2,3]
    # with open(way+'text.txt','w') as f:
    #     f.writelines(str(arr))
    # print(arr)
    
    
    
    
    
    