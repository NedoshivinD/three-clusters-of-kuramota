import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import linalg as LA
import time
import random

sp = [1.8234765819369751, 3.6469531638739503, 4, 2, 1.6444000000000014, 0.95]
N = 10

K = sp[2]
M = sp[3]
alpha = sp[4]
m = sp[5]

max_time = 2000
eps_map = 1e-3
eps = 1e-3
# 2.1043999999999996, 2.3899999999999926, 'default', 4.459708725242611, 2.636232143305636
times = np.linspace(0,max_time,1000)
omega = 1
start_fi1 = 1
def syst(t,param):
        x,y,v,w = param #[x,y,w,v] - точки
        f = np.zeros(4)
        # x with 1 dots
        f[0] = v
        # y with 1 dots
        f[1] = w
        # x with 2 dot
        f[2] = 1/m*(1/N * ((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha)) - v)
        # y with 2 dot
        f[3] = 1/m*(1/N * ((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-M*np.sin(y-x-alpha)) - w)
        
        return f

def syst2(t,x0,m,alpha):
        fi1,fi2,fi3,v,w,u = x0 #[x,y,w,v] - точки
        f = np.zeros(6)
        # fi1 with 1 dots
        f[0] = v
        # fi2 with 1 dots
        f[1] = w
        # fi3 with 1 dots
        f[2] = u
        # fi1 with 2 dot
        f[3] = 1/m*(omega + 1/N * ((- K)*np.sin(alpha) + M*np.sin(fi2 - fi1 - alpha) + (N-M-K)*np.sin(fi3 - fi1 - alpha)) - v)
        # fi2 with 2 dot
        f[4] = 1/m*(omega + 1/N * (-M*np.sin(alpha) + K*np.sin(fi1 - fi2 - alpha) + (N-M-K)*np.sin(fi3 - fi2 - alpha)) - w)
        # fi3 with 3 dot
        f[5] = 1/m*(omega + 1/N * (-(N-M-K)*np.sin(alpha) + K*np.sin(fi1 - fi3 - alpha) + M*np.sin(fi2 - fi3 - alpha)) - u)
        
        return f

def plot():
        start_point = [sp[0], sp[1],1e-3,1e-3]

        tmp = solve_ivp(syst, [0,max_time], start_point, t_eval=times)

        # plt.plot(tmp.t,np.angle(np.exp(1j*tmp.y[0])),label='X = ' + r'($\varphi_1 - \varphi_2$) mod 2$\pi$') #np.angle(np.exp(1j*tmp.y[0]))
        # plt.plot(tmp.t,np.angle(np.exp(1j*tmp.y[1])),label='Y = ' + r'($\varphi_1 - \varphi_3$) mod 2$\pi$', linestyle = '--') #np.angle(np.exp(1j*tmp.y[1]))
        plt.plot(tmp.t,tmp.y[0],label='X = ' + r'$\varphi_1 - \varphi_2$') #np.angle(np.exp(1j*tmp.y[0]))
        plt.plot(tmp.t,tmp.y[1],label='Y = ' + r'$\varphi_1 - \varphi_3$', linestyle = '--') #np.angle(np.exp(1j*tmp.y[1]))
        plt.xlim(0, max_time)
        # plt.ylim(-4, 4)
        plt.xlabel('t')
        # plt.ylabel(r'($\varphi_1 - \varphi_i$) mod 2$\pi$')
        plt.ylabel(r'$\varphi_1 - \varphi_i$')
        plt.legend()
        plt.grid()
        plt.show()

def order_parameter(t,arr):
        res = []
        for i in range(len(t)):
            sumr = 0
            sumi = 0                
            for j in range(3):
                tmp = np.exp(arr[:,i][j]*1j)
                sumr += tmp.real
                sumi += tmp.imag
                sum = 1/3 * np.sqrt(sumr ** 2 + sumi ** 2)
            res.append(sum)
        return res


def rec_dinamic(params,m,alpha):
        start_point=np.zeros(6)
        start_point[0],start_point[1],start_point[2] = params 
        start_point[0] += eps
        start_point[1] += eps
        start_point[2] += eps
        tmp = solve_ivp(syst2, [0, max_time],start_point, t_eval=times,args=(m,alpha))
        return tmp

def plot_R1():
        arr = rec_dinamic([0,sp[0],sp[1]],m,alpha)
        # plt.plot(arr.t,arr.y[0]-arr.y[1])
        # plt.plot(arr.t,arr.y[0]-arr.y[2])
        # plt.show()
        R1 = order_parameter(arr.t,arr.y)
        plt.plot(arr.t,R1)
        plt.xlabel('t')
        plt.ylabel('R1')
        plt.ylim(0,1.1)
        plt.show()

def jakobi(param):
        x,y,m,alpha = param
        
        f = []
        f.append([-1/m, 0, -(M*np.cos(alpha + x) - np.cos(alpha - x + y)*(K + M - N) + K*np.cos(alpha - x))/(N*m),
             -(np.cos(alpha - x + y)*(K + M - N) - np.cos(alpha + y)*(K + M - N))/(N*m)])
        f.append([0, -1/m, (M*np.cos(alpha + x - y) - M*np.cos(alpha + x))/(N*m),
            -(M*np.cos(alpha + x - y) - np.cos(alpha + y)*(K + M - N) + K*np.cos(alpha - y))/(N*m)])
        f.append([1.0, 0, 0, 0])
        f.append([0, 1.0, 0, 0])
        
        arr = np.array(f)
        return(arr)

    #поиск собственных чисел при определененных параметрах
def eigenvalues(param):
        matrix = jakobi(param)
        lam, vect = LA.eig(matrix)
        return lam
def jacobi_full(start_point):
        phi = np.zeros(N)
        v = np.zeros(N)
        for i in range(N):
            phi[i] = start_point[i]
            v[i] = start_point[i+N]

        derivatives = np.array([])
        for i in range(N):
            tmp_arr = np.zeros(N)
            for j in range(N):
                sum = 0
                if j == i:
                    for k in range(N):
                        if k == j:
                            continue
                        else:
                            sum+=np.cos(phi[k]-phi[j]-alpha)
                    tmp_arr[j] = - 1/(m*N) * sum
                else:
                    tmp_arr[j] = 1/(m*N) * np.cos(phi[j]-phi[i]-alpha)

            derivatives = np.append(derivatives,tmp_arr)
        derivatives = derivatives.reshape((N,N))

        eye = np.eye(N)
        block1 = -1/m * eye
        block2 = derivatives
        block3 = eye
        block4 = np.zeros((N,N))
        res = np.block([[block1,block2],[block3,block4]])
        # print(res)
        return res
def anti_zamena_2(arr):
        fi1 = start_fi1
        fi2 = fi1 - arr[0]
        fi3 = fi1 - arr[1]
        ress = [fi1, fi2, fi3, int(arr[2]), int(arr[3])]
        return ress

def eigenvalues_full(param):
        start_point = np.zeros(2*N)
        par = anti_zamena_2(arr=param)
        phi1,phi2,phi3,m,alpha = par
        v1,v2,v3 = (0,0,0)
        for i in range(K):
            start_point[i] = phi1
            start_point[i+N] = v1
        for i in range(M):
            start_point[K+i] = phi2
            start_point[K+i+N] = v2
        for i in range(N-M-K):
            start_point[K+M+i] = phi3
            start_point[K+M+i+N] = v3
        matrix = jacobi_full(start_point)
        lam, vect = LA.eig(matrix)
        return lam

def plot_lams(param):
        old_lam = eigenvalues(param)
        new_lam = eigenvalues_full(param)
        print(new_lam)
        old_lam = np.array(old_lam)
        new_lam = np.array(new_lam)
        real_deal_old = old_lam.real
        not_real_deal_old = old_lam.imag
        real_deal_new = new_lam.real
        not_real_deal_new = new_lam.imag

        plt.scatter(real_deal_old, not_real_deal_old, c='b', marker='o', label='собственные числа редуцированной системы')
        plt.scatter(real_deal_new, not_real_deal_new, c='r', marker='x', label='собственные числа полной системы')
        plt.grid()
        plt.legend()
        plt.xlabel(r'Re($\lambda$)')
        plt.ylabel(r'Im($\lambda$)')
        plt.show()


 #тепловая карта----------------------------------------------------------------------------------
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
                s += np.sin(phi_i - phi[j]  + alpha)

            f[j] = v[j]
            f[j+len(phi)]=(s/N + w - v[j])/m

            s = 0
        return f
        
def din_thr_map(phi,v,par,t,t_max):
        start_point = np.zeros(len(phi)*2)
        for i in range(len(phi)):
            start_point[i] = phi[i]
            start_point[i+len(phi)] = v[i]

        res = solve_ivp(iter_din,[0,t_max],start_point, args=[par],t_eval=t,rtol= 10e-10,atol=10e-10) # t_eval=t,
        
        return res.y

def up_arr(start_phi, arr,N,num_elems):
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
            res = np.append(res,tmp+random.uniform(-1, 1)*1e-10)
        
        tmp-= start_phi
        
        for i in range(len(razb[1:3])):
            tmp = tmp+arr[i]
            for j in range(razb[i+1]):
                res = np.append(res, start_phi -tmp+random.uniform(-1, 1)*1e-10)
            tmp = tmp-arr[i]
        return res

def plot_warm_map(param, count):
        phi,eps,alpha,t_max = param
        
        v = np.zeros(len(phi))
        v = v + 1e-3
        
        w = 1
        t = np.linspace(0,t_max,t_max)
        a = din_thr_map(phi,v,[alpha,w],t,t_max)
        
        matrix = np.array([])
        for i in range(len(phi)):
            matrix = np.append(matrix,a[i])
        
        matrix = matrix.reshape((len(phi),len(matrix)//len(phi)))

        l = len(phi) - 1
        while l>=0:
            matrix[l] = matrix[l] - matrix[0]
            l-=1

        fig = plt.figure(figsize=(10,4))
        matrix = np.angle(np.exp(1j*matrix))
        p = plt.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect='auto')
        fig.colorbar(p,label=r'($\varphi_1 - \varphi_i$) mod $2\pi$')
        plt.xlabel("t")
        plt.ylabel("N")
        plt.show()
        plt.close(fig)
        plt.clf()


    #тепловая карта---------------------------------------------------------------------------------------------------

def work(alpha):
    x = [sp[0], sp[1], K, M, alpha] #alpha = 2,m=10 
    tmp_count = 0
    start_phi = 1
    phi1 = up_arr(start_phi,x,N,N)
    alpha = x[4]
    t_amx = max_time    
    # phi1 = [2.474646, 2.474646, 2.474646, 0, 0]
    # alpha = 0
    
        
    # print(phi1)
    # print(str(tmp_count+1)+":")
    
    plot_warm_map([phi1,eps,alpha,t_amx],tmp_count)


plot()
plot_R1()
work(alpha)
plot_lams([sp[0],sp[1],m,alpha])