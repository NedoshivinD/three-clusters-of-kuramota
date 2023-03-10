import numpy as np
import threading
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

n_job = 4

def iter_din(t, start_point , par):
    
    alpha,w = par
    for i in range(len(start_point)//2):
        phi[i] = start_point[i]
        v[i] = start_point[i+len(phi)]
    # phi, v = start_point
    s = 0
    
    
    f = np.zeros(len(phi)*2)
    
    for j in range(len(phi)):
        for phi_i in phi:
            s += np.sin(phi_i - phi[j] + alpha)
        
        f[j] = s + w + v[j]
        
        f[j+len(phi)] = v[j]
        
        
    return f
        
    
    

def din_thr_map(phi,v,par,t,t_max):
    res = []
    start_point = np.zeros(len(phi)*2)
    for i in range(len(phi)):
        start_point[i] = phi[i]
        start_point[i+len(phi)] = v[i]
        
    
    res = solve_ivp(iter_din,[0,t_max],start_point, args=[par],t_eval=t,rtol= 10e-7,atol=10e-7)
    
    
    
    
    return res

def iter_din_2(phi, v, par, ur, tmp_phi, res, time, event_for_wait, event_for_set):
    
    for i in range(time):
            
        event_for_wait.wait() # wait for event
        event_for_wait.clear() # clean event for future
        
        alpha,w = par
        s = 0
        
        for phi_i in phi:
            s += np.sin(phi_i - phi[ur] + alpha)
        
        
        tmp_phi[ur] = s + w + v[ur]
        
        if ur == len(phi)-1:
            for j in range(len(phi)):
                phi[j] = tmp_phi[j]
            res.append(phi)

        event_for_set.set() # set event for neighbor thread


def din_thr_map2(phi,v,par,time):
    res = []
    tmp_phi = []
    for i in range(len(phi)):
        tmp_phi.append(0)
    
    e = []
    for i in range(len(phi)):
        e.append(threading.Event())
    
    t = []

    for j in range(len(phi)):
        if j != len(phi)-1:
            t.append(threading.Thread(target=iter_din_2, args=(phi, v, par, j, tmp_phi, res, time, e[j], e[j+1])))
        else:
            t.append(threading.Thread(target=iter_din_2, args=(phi, v, par, j, tmp_phi, res, time, e[j], e[0])))
    
    for i in range(len(phi)):
        t[i].start()

    e[0].set()

    for i in range(len(phi)):
        t[i].join()
        
    
    
    return res    

    
if __name__ == "__main__":
    phi = [0, np.pi, np.pi, np.pi, np.pi, np.pi] #0,0,0,0,0,0,0,0, -np.pi, -np.pi,-np.pi, -np.pi,-np.pi, -np.pi, np.pi,
    v = np.zeros(len(phi))
    alpha = 0
    w = 1
    t_max = 1000
    t = np.linspace(0,t_max,t_max)
    a = din_thr_map(phi,v,[alpha,w],t,t_max)
    
    matrix = np.array([])
    for i in range(len(phi)):
        matrix = np.append(matrix,a.y[i])
    
        
    for i in range(len(matrix)):
        matrix[i] = math.remainder(matrix[i], 2*math.pi)
    matrix = matrix.reshape((len(phi),t_max))
    
    for i in range(len(phi)):
        matrix[i] = matrix[i] - matrix[0]
    
    
    # plt.xlim((0,20))
    plt.imshow(matrix, cmap ='hot', interpolation='nearest')
    plt.show()
    # plt.savefig('tmp.svg')
    
    # print(din_thr_map2(phi,v,[alpha,w],t))
    
    # res = []
    # for i in range(3):
    #     iter_din(phi,v,[alpha,w],i,res)
    # print(res)

