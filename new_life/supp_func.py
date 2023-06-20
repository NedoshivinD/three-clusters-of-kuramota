import numpy as np
import threading
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from supp_func import *
import random

def razb_config():
    way = 'new_life\\config.txt'
    res = []
    tmp = ''
    with open(way,'r') as f:
        for c in f.readline():
            if c==',':
                res.append(int(tmp))
                tmp = ''
                continue
            tmp+=c
    res.append(int(tmp))
    return res

def cm_to_inch(value):
    return value/2.54

def razb_text(way):
    res = []
    elem = []
    value = ''
    f_htrix = False
    f_kvadr = False

    with open(way+'text.txt','r') as f:
        for line in f.readlines():
            for c in line:
                if c == ' ':
                    continue
                if c == '[':
                    f_kvadr = True
                    continue
                if c == ']':
                    f_kvadr = False
                    if value != '':
                        elem.append(float(value))
                    if len(elem) != 0:
                        res.append(elem)
                        f_htrix = False
                    value = ''
                    elem = []
                    continue
                if f_kvadr:
                    if c==',':
                        if value != '':
                            if f_htrix:
                                elem.append(value)
                            else:
                                elem.append(float(value))
                            value = ''
                        continue
                    if c=='\'':
                        f_htrix = True
                        continue
                    value+=c

    return res

def get_ind_text(arr,point):
    alpha,beta = point
    eps = 0.05
    ind = None
    for elem in arr:
        if np.abs(elem[0]-alpha)<eps and np.abs(elem[1]-beta)<eps:
            ind = arr.index(elem)
            break
    return ind

def add_to_good_arr(good_arr,arr):
    tmp= good_arr.copy()
    for line in arr:
        for elem in line:
            if elem !=[]:
                tmp.append(elem)
    return tmp

def add_unic_good_arr(good_arr,arr):
    tmp= good_arr.copy()
    f = True
    for a in arr:
        for g_a in tmp:
            if a[:2] == g_a[:2]:
                f = False
                break
        if f:
            tmp.append(a)
        f = True
    return tmp

def get_good_arr(arr):
    good_arr = []
    for line in arr:
        for elem in line:
            if elem !=[]:
                good_arr.append(elem)
    return good_arr

#карта------------------------------------------------------
def iter_din(t, start_point , par):
    alpha,m,w = par
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
            s += np.sin(phi_i - phi[j]  + alpha)
        
        f[j] = v[j]
        f[j+len(phi)]=(s/lens + w - v[j])/m

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
    
    razb = [int(arr[2]),int(arr[3]),int(N-arr[2]-arr[3])]
    
    for i in range(len(arr[0:2])):
        if np.pi - np.abs(arr[i]) < 1e-5 and np.pi - np.abs(arr[i]) > -1e-5:
            arr[i] = np.pi * np.sign(arr[i])
    if np.pi - np.abs(arr[4])<1e-5:
        arr[4] = np.pi * np.sign(arr[4])
    
    tmp +=start_phi

    for i in range(razb[0]):
        res = np.append(res,tmp+random.uniform(-1, 1)*1e-2)
    
    tmp-= start_phi
    
    for i in range(len(razb[1:3])):
        tmp = tmp+arr[i]
        for j in range(razb[i+1]):
            res = np.append(res, start_phi - tmp + random.uniform(-1, 1)*1e-2)
        tmp = tmp-arr[i]
    return res
    
def work(param,K,M,show):
    
    phi,eps,alpha,m,t_max = param
    
    v = np.zeros(len(phi))
    v = v + 1e-1
    
    w = 1
    t = np.linspace(0,t_max,1000)
    a = din_thr_map(phi,v,[alpha,m,w],t,t_max)
    
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
    if show==1:
        fig = plt.figure(figsize=(10,4))
        matrix = np.angle(np.exp(1j*matrix))
        # print(matrix)
        p = plt.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect='auto')
        fig.colorbar(p,label=r'($\varphi_1 - \varphi_i$) mod $2\pi$')
        plt.xlabel("t")
        plt.ylabel("N")
        plt.show()

    return analyse_matrix(matrix,K,M,show)
    
    
    
def heat_map(par,show):
    tmp = razb_config()
    eps = 1e-5
    low_arr = par
    start_phi = 1
    eps = 1e-7
    phi1 = up_arr(start_phi,low_arr,tmp[0],tmp[0],eps)
    alpha = low_arr[4]
    m = low_arr[5]
    
    t_amx = 500
    
    return work([phi1,eps,alpha,m,t_amx],low_arr[2],low_arr[3],show)

def analyse_matrix(matrix,K,M,show):
    ind = 0
    num_klust = []
    while ind < len(matrix):
        tmp_razn = matrix[ind][-1]
        f = True
        for i,elem in enumerate(num_klust):
            if np.abs(tmp_razn - elem[0]) < 1e-1:
                num_klust[i][1]+=1
                f = False
                break
        if f:
            num_klust.append([tmp_razn,1])
        ind+=1

    maxima = matrix[0][0]
    minima = matrix[0][0]
    f_min = False
    f_max = False

    if show == 1:
        print(num_klust)

    if len(num_klust)==3 or len(num_klust)==2:  
        for line in matrix:
            mini_arr = line[int(-len(line)/10):]
            max_mini_arr = max(mini_arr)
            min_mini_arr = min(mini_arr)
            if np.abs(max_mini_arr - min_mini_arr) > 0.5 and np.abs(max_mini_arr - min_mini_arr) < 3:
                return "koleb"
        for line in matrix:
            for elem in line[int(-len(line)/10):]:
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
        if  num_klust[0][1]==K and num_klust[1][1]==M:
            return "same_mode"
        else:
            return "new_mode"
    elif len(num_klust)==1:
        return "in-phase"
            
    return "default"


if __name__ == "__main__":
    # arr1 = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
    # arr2 = [[1,2,10],[2,3,4],[4,3,5],[4,5,6]]

    # f = True
    # for a2 in arr2:
    #     for a1 in arr1:
    #         if a1[:2] == a2[:2]:
    #             f = False
    #             break
    #     if f:
    #         arr1.append(a2)
    #     f = True
    # print(arr1)

    tmp = razb_config()
    param = [4.459709, 2.636232, 2, 1, 2.0944, 1]

    
    way= f"new_life\\res\\n_{tmp[0]}\\tmp\\{param}\\"
    
    arr = razb_text(way)
    ind = get_ind_text(arr,[0.2,2.2])
    if ind==None:
        print("not in array")
    else:
        print(arr[ind])

    













# point_analyse 
# par = self.anti_par_zam(params)
        # par = np.reshape(par, (len(par[0])))

        # start_point=np.zeros(4)
        # start_point[0],start_point[1] = par[0:2] 
        # start_point[0] = start_point[0]+eps
        # start_point[1] = start_point[1]+eps
        # start_point[2] = eps
        # start_point[3] = eps

        # return self.__ord_par_tong__(start_point,show,t)
