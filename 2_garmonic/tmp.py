import numpy as np
import threading
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

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
        res = np.append(res,tmp+eps)
    
    tmp-= start_phi
    
    tmp = tmp+arr[0]
    
    for i in range(razb[1]):
        res = np.append(res, start_phi - tmp)
        
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
    plt.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect=4)
    plt.show()
    
    
    # plt.savefig('tmp.svg')
    
def heat_map(par):
    low_arr = par
    start_phi = 1
    eps = 1e-7
    phi1 = up_arr(start_phi,low_arr,5,5,eps)
    alpha = low_arr[2]
    beta = low_arr[3]
    k1 = low_arr[4]
    k2 = low_arr[5]
    
    t_amx = 10000
    
    work([phi1,eps,alpha,beta,k1,k2,t_amx])
    
if __name__ == "__main__":
    
    eps = 0#1e-5
    #[3.141593, 0.0, 2, 2, 3.14159]         40 неуст
    #[3.141593, 3.141593, 3, 1, 3.14159]    4  уст
    low_arr = [2.094395, 2, 3.14159, 3.14159, 1, 1]
                                                        #0 0 0 2.474646 2.474646              0 0 2.474646 2.474646 0
    start_phi = 1
    eps = 1e-1
    phi1 = up_arr(start_phi,low_arr,5,5,eps)
    # phi1[3] +=0.000001
    alpha = low_arr[2]
    beta = low_arr[3]
    k1 = low_arr[4]
    k2 = low_arr[5]
    t_amx = 10000
    
    # phi1 = [2.474646, 2.474646, 2.474646, 0, 0]
    # alpha = 0
    
        
    # print(phi1)
    work([phi1,eps,alpha,beta,k1,k2,t_amx])
    
    
    
    
    
    