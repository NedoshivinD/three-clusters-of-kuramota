import numpy as np
import threading
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

n_job = 4

def iter_din(t, start_point , par):
    
    alpha,w = par
    phi = np.zeros(len(start_point)//2)
    v = np.zeros(len(start_point)//2)
    
    for i in range(len(start_point)//2):
        phi[i] = start_point[i]
        v[i] = start_point[i+len(phi)]

    s = 0
    
    
    f = np.zeros(len(phi)*2)
    
    for j in range(len(phi)):
        for phi_i in phi:
            s += np.sin(phi_i - phi[j] + alpha)
        
        f[j] = round(s + w + v[j], 7)
        
        f[j+len(phi)] = round(v[j], 7)
        
        
    return f
        
def din_thr_map(phi,v,par,t,t_max):
    res = []
    start_point = np.zeros(len(phi)*2)
    for i in range(len(phi)):
        start_point[i] = phi[i]
        start_point[i+len(phi)] = v[i]
        
    
    res = solve_ivp(iter_din,[0,t_max],start_point, args=[par],rtol= 10e-13,atol=10e-13) # t_eval=t,
    
    
    
    
    return res

def up_arr(arr,N,num_elems):
    res = np.array([])
    tmp = np.zeros(num_elems//N)
    
    if N>num_elems:
        num_elems = N
    
    razb = [arr[2],arr[3],N-arr[2]-arr[3]]
    
    for i in range(razb[0]):
        res = np.append(res,tmp)
    
    for i in range(len(razb[1:3])):
        tmp = tmp+arr[i]
        for j in range(razb[i+1]):
                    
            res = np.append(res,tmp)
        tmp = tmp-arr[i]
    return res
    
def work(param):
    
    phi,eps,alpha,t_max = param
    
    for i in range(len(phi)-len(phi)//2):#
        phi[i] += eps
    
    v = np.zeros(len(phi))
    w = 1
    t = np.linspace(0,t_max,t_max)
    a = din_thr_map(phi,v,[alpha,w],t,t_max)
    
    matrix = np.array([])
    for i in range(len(phi)):
        matrix = np.append(matrix,a.y[i])
    

    matrix = np.angle(np.exp(1j*matrix))
    matrix = matrix.reshape((len(phi),len(matrix)//len(phi)))

    l = len(phi) - 1
    while l>=0:
        # if l != 19:
        matrix[l] = matrix[l] - matrix[0]
        l-=1

    
    print(matrix)
    
    plt.imshow(matrix, cmap ='hot', interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect=4)
    plt.show()
    
    
    # plt.savefig('tmp.svg')
    

if __name__ == "__main__":
    
    eps = 0# 1e-17
    
    low_arr = [1.823477, 3.646953, 1, 2, 2.0944]
    
    phi1 = up_arr(low_arr,5,20)
    alpha = low_arr[4]
    t = 10000
    
    # phi = [0, 0, 0, 0, 1.823477, 1.823477, 1.823477, 1.823477, 1.823477, 1.823477, 3.646953, 3.646953, 3.646953, 3.646953, 3.646953, 3.646953] #[0,0,0,0, 1.823477, 1.823477,1.823477, 1.823477,1.823477, 1.823477, 3.646953, 3.646953, 3.646953, 3.646953, 3.646953, 3.646953] 
    # alpha = 0
    
        
    # print(phi1)
    work([phi1,eps,alpha,t])
    
    
    
    
    
    