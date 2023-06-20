import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import linalg as LA
import joblib 

N = 9
K = 4
M = 1
# alpha = 0
# m =  5

max_time = 1000
max_m = 10
eps = 1e-3
sp = [1.696124, 3.392248]
omega = 1

time = np.linspace(0,max_time,100)

def syst(t,x0,m,alpha):
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

def syst2(t,param,m,alpha):
        
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

def dinamic(params,m,alpha):
    start_point=np.zeros(4)
    start_point[0],start_point[1] = params 
    tmp = solve_ivp(syst2, [0, max_time],start_point, t_eval=time,args=(m,alpha))
    
    return tmp

def rec_dinamic(params,m,alpha):
        start_point=np.zeros(6)
        start_point[0],start_point[1],start_point[2] = params 
        start_point[0] += eps
        start_point[1] += eps
        start_point[2] += eps
        tmp = solve_ivp(syst, [0, max_time],start_point, t_eval=time,args=(m,alpha))
        return tmp

def func_for_paral(m,alpha):
        res_arr = []
        # arr = rec_dinamic([0,-sp[0],-sp[1]],m,alpha)
        # plt.plot(arr.t,arr.y[0]-arr.y[1])
        # plt.plot(arr.t,arr.y[0]-arr.y[2])
        # plt.show()
        arr = dinamic([sp[0],sp[1]],m,alpha)
        # R1 = order_parameter(arr.t,arr.y)
        # plt.plot(arr.t,R1)
        # plt.show()
        new_arr_x = []
        for i in range(len(arr.y[0])//2,len(arr.y[0])):
            new_arr_x.append(arr.y[0][i])
        max_x = max(new_arr_x)
        min_x = min(new_arr_x)
        if (abs(max_x - min_x)<1e-3):
            res_arr.append([m,alpha,0])
        else: res_arr.append([m,alpha,1])

        # lam = eigenvalues([sp[0],sp[1],m,alpha])
        # flag = 0
        # for i in range(len(lam)):
        #     if lam[i].imag != 0:
        #         flag = 1
        # if flag == 0:
        #     res_arr.append([m,alpha,0])
        # else:
        #     res_arr.append([m,alpha,1])
        
        return res_arr

def paral():
        res_arr = []
        m_arr = np.linspace(1,max_m,100)
        al_arr = np.linspace(0,2*np.pi,100)
        # m = 5
        # alpha = 0
        res_arr = joblib.Parallel(n_jobs = 6)(joblib.delayed(func_for_paral)(m,alpha) for m in m_arr for alpha in al_arr)
        # ress_arr = func_for_paral(m,alpha)
        res_arr = np.array(res_arr)
        res_arr = res_arr.T
        # print(np.shape(res_arr))
        res_arr = np.reshape(res_arr,(3,10000))
        first_border = []
        second_border = []
        for i in range(len(res_arr[0])-1):
            if res_arr[2][i] == 0 and res_arr[2][i+1] == 1:
                first_border.append(res_arr[1][i+1])
                continue
            if res_arr[2][i] == 1 and res_arr[2][i+1] == 0:
                second_border.append(res_arr[1][i])
                continue
        # first_border_arr = np.linspace(first_border,first_border,len(m_arr))
        # second_border_arr = np.linspace(second_border,second_border,len(m_arr))
        fig, ax = plt.subplots()
        # ax.plot(first_border,m_arr,c='black')
        # ax.plot(second_border,m_arr,c='black')

        for i in range(len(res_arr[0])):
            if res_arr[2][i]==1:
                # print(res_arr[1][i],res_arr[0][i])
                plt.scatter(res_arr[1][i],res_arr[0][i],c='r',alpha=0.4)
            else:
                plt.scatter(res_arr[1][i],res_arr[0][i],c='b',alpha=0.4) 
                # print(res_arr[1][i],res_arr[0][i])

        # ax.axvline(x=first_border_arr[0],c='black')
        # ax.axvline(x=second_border_arr[0],c='black')
        
        # ax.axvspan(first_border, second_border,alpha=0.8, color='red')
        # ax.axvspan(0, first_border,alpha=0.8, color='b')
        # ax.axvspan(second_border, 2*np.pi,alpha=0.8, color='b')
        # ax.axhspan(first_border_arr[0], second_border_arr[0],m_arr[0],m_arr[-1], alpha=0.8, color='red')
        
        
        
        plt.xlim(0,2*np.pi)
        plt.ylim(m_arr[0],m_arr[-1])
        plt.xlabel(r'$\alpha$')
        plt.ylabel('m')
        
        plt.show()

paral()
