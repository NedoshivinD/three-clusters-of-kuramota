import numpy as np
import threading
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from supp_func import *
import random





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

    res = solve_ivp(iter_din,[0,t_max],start_point,t_eval=t, args=[par])#,rtol= 10e-10,atol=10e-10) # 
    
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
    
def work(param,M,show):
    
    phi,eps,alpha,beta,k1,k2,t_max = param
    
    v = np.zeros(len(phi))
    v = v + 1e-1
    
    w = 1
    t = np.linspace(0,t_max,1000)
    a = din_thr_map(phi,v,[alpha,beta,k1,k2,w],t,t_max)
    
    matrix = np.array([])
    for i in range(len(phi)):
        matrix = np.append(matrix,a[i])
    
    matrix = matrix.reshape((len(phi),len(matrix)//len(phi)))

    l = len(phi) - 1
    while l>=0:
        matrix[l] = matrix[l] - matrix[0]
        l-=1

    matrix+=eps
    
    matrix = np.angle(np.exp(1j*matrix))
    # print(matrix)

    if show == 1:
        fig, ax = plt.subplots()
        
        p = ax.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect='auto')
        fig.colorbar(p,label=r'$\varphi_1 - \varphi_i$')
        plt.xlabel("t")
        plt.ylabel("N")
        plt.show()
    return analyse_matrix(matrix,M)
    
    
    
def heat_map(par,show=1):
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
    
    t_amx = 100
    
    return work([phi1,eps,alpha,beta,k1,k2,t_amx],low_arr[1],show)

def analyse_matrix(matrix,M):
    maxima = matrix[0][0]
    minima = matrix[0][0]
    f_min = False
    f_max = False
    for line in matrix[-10:]:
        for elem in line:
            if maxima < elem:
                maxima = elem
                if np.abs(maxima) > 3:
                    f_max = True
            if minima > elem:
                minima = elem
                if np.abs(minima) > 3:
                    f_min = True
            if f_min and f_max:
                return "vrash"

    ind = 0
    num_klust = []
    while ind < len(matrix):
        tmp_razn = matrix[0][-1] - matrix[ind][-1]
        f = True
        for i,elem in enumerate(num_klust):
            if np.abs(tmp_razn - elem[0]) < 1e-2:
                num_klust[i][1]+=1
                f = False
                break
        if f:
            num_klust.append([tmp_razn,1])
        ind+=1

    if len(num_klust)==2:
        if  num_klust[0][1]==M:
            return "two"
        else :
            return "two_2"
            
    return "default"
    
    
    
    


            
n_job = 8
eps_analyse_map = 1e-5

if __name__ == "__main__":
    

    tmp = razb_config()
    eps = 1e-5

    low_arr = [87.96459430051421, 2, 1.1500000000000008, 1.8444000000000016, 1, 1]
    # low_arr = [ 1.6696164961860338, 2, 0.04, 2.044400000000001, 1, 1]
    
    start_phi = 1
    eps = 1e-1
    phi1 = up_arr(start_phi,low_arr,tmp[0],tmp[0],eps)
    alpha = low_arr[2]
    beta = low_arr[3]
    k1 = low_arr[4]
    k2 = low_arr[5]
    t_amx = 100

    print(work([phi1,eps,alpha,beta,k1,k2,t_amx],low_arr[1],1))
    
    
    
    
    
    