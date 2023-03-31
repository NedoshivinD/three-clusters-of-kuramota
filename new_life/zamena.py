import numpy as np
from requests import patch
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root
import joblib 
from numpy import linalg as LA
import os, shutil
from itertools import groupby
from scipy.integrate import solve_ivp


#Параметры системы 

N_JOB = 4
col_razb = 10
MAX_GRAPH = 50
eps = 0.1



class Equilibrium_states(object):
    
    #инициализация системы
    def __init__(self,p = [3,1]):
        self.N, self.m  = p
        self.M = 0
        self.K = 0
        self.alpha = 0
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,100,100)

    # Сама система (возможно стоит использовать при расчете состояний равновесия)
    def syst(self,t,param):
        N = self.N
        M = self.M
        K = self.K
        alpha = self.alpha
        m = self.m
        
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
    
    def __trash_off__(self,arr):
        tmp = []
        tmp_res = []
        res = []
        par = arr[0][2:5]
        for i in arr:
            if par == i[2:5]:
                tmp = [round(np.sin(i[0])+np.cos(i[0])),round(np.sin(i[1])+np.cos(i[1]))]
                if tmp not in tmp_res:
                    tmp_res.append(tmp)
                    res.append(i)
            else :
                tmp_res = []
                par = i[2:5]
                tmp = [round(np.sin(i[0])+np.cos(i[0])),round(np.sin(i[1])+np.cos(i[1]))]
                tmp_res.append(tmp)
                res.append(i)
        return res

    #поиск состояний равновесия для определенного параметра
    def state_equil(self,NKMA=[3,1,1,np.pi/2]):
        all_sol = [] 
        all_sol_full = []
        self.N,self.K,self.M,self.alpha = NKMA
        ran_x=[0,2*np.pi-0.001]
        ran_y=[0,2*np.pi-0.001]
        
        X = np.linspace(ran_x[0],ran_x[1],col_razb)
        Y = np.linspace(ran_y[0],ran_y[1],col_razb)
        v = 0
        w = 0
        
        for x in X:
            for y in Y:
                sol = root(self.syst,[x,y,v,w],args=(0),method='lm')
                xar,yar,var,war = sol.x
                xar = round(xar,6)
                yar = round(yar,6)
                if (xar>=0 and xar<2*np.pi) and (yar<2*np.pi and yar>=0):
                    if [round(xar,3),round(yar,3),self.K,self.M,round(self.alpha,5)] not in all_sol:
                        if round(xar,2) == round(2*np.pi,2) and  round(yar,2) != round(2*np.pi,2):
                            all_sol_full.append([0.0,yar,self.K,self.M,round(self.alpha,5)])
                            all_sol.append([0.0,round(yar,3),self.K,self.M,round(self.alpha,5)])
                        elif round(yar,2) == round(2*np.pi,2) and round(xar,2) != round(2*np.pi,2):
                            all_sol_full.append([xar,0.0,self.K,self.M,round(self.alpha,5)])
                            all_sol.append([round(xar,3),0.0,self.K,self.M,round(self.alpha,5)])
                        elif (round(xar,2) == round(2*np.pi,2) and round(yar,2) == round(2*np.pi,2)):
                            all_sol_full.append([0.0,0.0,self.K,self.M,round(self.alpha,5)])
                            all_sol.append([0.0,0.0,self.K,self.M,round(self.alpha,5)])
                        else:
                            all_sol_full.append([xar,yar,self.K,self.M,round(self.alpha,5)])
                            all_sol.append([round(xar,3),round(yar,3),self.K,self.M,round(self.alpha,5)])
        
        new_arr = self.__trash_off__(all_sol_full)
        # new_arr = [el for el, _ in groupby(all_sol)]
        return new_arr
    
    #поиск всех состояний равновесия
    def parall_st_eq(self):
        N = self.N
        M = np.linspace(1,N-2,N-2, dtype = 'int')
        K = np.linspace(N-2,1,N-2, dtype = 'int')
        
        alpha = np.linspace(0,np.pi,4)
        self.sost = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.state_equil)([N,k,m,al]) for k in K for m in M for al in alpha if m+k<N)
        
        self.rec_st_and_unst_st_eq()
        
        self.save_all_sost()
        
    
    def save_all_sost(self):
        name = f"new_life\\res\\n_{self.N}\\res_n_{self.N}.txt"
        
        with open(name,"w",encoding="utf-8") as file: 
            for tmp in self.sost:
                for j in tmp:
                    j = str(j)+'\n'
                    file.write(j)  
                # file.write('--------------------------------------------------------\n')

    #матрица якоби
    def jakobi(self, param):
        x,y,K,M,alpha = param
        N = self.N
        m = self.m
        
        f = []
        f.append([-1/m, 0, -(M*np.cos(alpha + x) - np.cos(alpha - x + y)*(K + M - N) + K*np.cos(alpha - x))/(N*m),
             -(np.cos(alpha - x + y)*(K + M - N) - np.cos(alpha + y)*(K + M - N))/(N*m)])
        f.append([0, -1/m, (M*np.cos(alpha + x - y) - M*np.cos(alpha + x))/(N*m),
            -(M*np.cos(alpha + x - y) - np.cos(alpha + y)*(K + M - N) + K*np.cos(alpha - y))/(N*m)])
        f.append([1.0, 0, 0, 0])
        f.append([0, 1.0, 0, 0])

        # f.append([-1/m, 0, 1/(N*m)*(-M*np.cos(x+alpha) - K*np.cos(x - alpha) - (N - M - K)*np.cos(x-y-alpha)),
        #     1/(N*m)*(-(N-M-K)*np.cos(y + alpha) + (N-M-K)*np.cos(x-y-alpha))])
        # f.append([0, -1/m, 1/(N*m)*(-M*np.cos(x+alpha) + M*np.cos(y-x-alpha)),
        #     1/(N*m)*(-K*np.cos(y-alpha)-(N-M-K)*np.cos(y+alpha)-M*np.cos(y-x-alpha))])
        # f.append([1.0, 0, 0, 0])
        # f.append([0, 1.0, 0, 0])
        
        arr = np.array(f)
        return(arr)

    #поиск собственных чисел при определененных параметрах
    def eigenvalues(self,param):
        matrix = self.jakobi(param)
        lam, vect = LA.eig(matrix)
        return lam
    
    # def jacobi_full(self,start_point):
    #     N = self.N
    #     omega = self.omega
    #     m = self.m
    #     alpha = self.alpha
    #     phi = np.zeros(N)
    #     v = np.zeros(N)
    #     for i in range(N):
    #         phi[i] = start_point[i]
    #         v[i] = start_point[i+N]
    #     f = np.zeros(2*N)

    #     s = 0

    #     derivatives = np.array([])
    #     for i in range(N):
    #         tmp_arr = np.zeros(N)
    #         for j in range(N):
    #             sum = 0
    #             if j == i:
    #                 for k in range(N):
    #                     if k == j:
    #                         pass
    #                     else:
    #                         sum+=np.cos(phi[k]-phi[j]+alpha)
    #                 tmp_arr[j] = - 1/(m*N) * sum
    #             else:
    #                 tmp_arr[j] = 1/(m*N) * np.cos(phi[j]-phi[i]+alpha)

    #         derivatives = np.append(derivatives,tmp_arr)

    #     # eye = np.eye(N)
    #     # block1 = -1/m * eye
    #     # block2 = derivatives
    #     # block3 = eye
    #     # block4 = np.zeros((N,N))
    #     # res = np.block([[block1,block2],[block3,block4]])
    #     # return res

    #     res = np.array([])
    #     for i in range(N):
    #         string_arr = np.zeros(2*N)
    #         string_arr[i] = - 1 / m
    #         for j in range(N):
    #             string_arr[N+j] =  derivatives[i+j]
    #         res=np.append(res, string_arr)
    #     for i in range(N):
    #         string_arr = np.zeros(2*N)
    #         string_arr[i] = 1
    #         res=np.append(res, string_arr)
    #     res = res.reshape(2*N,2*N)
    #     print(res)
    #     return res
    
    # def eigenvalues_full(self,start_point):
    #     matrix = self.jacobi_full(start_point)
    #     lam, vect = LA.eig(matrix)
    #     return lam

    #Сохранение устойчивых и неустойчивых состояний равновесия 
    def rec_st_and_unst_st_eq(self):
        name_s = f"new_life\\res\\n_{self.N}\\stable_{self.N}.txt"
        name_n = f"new_life\\res\\n_{self.N}\\non_stable_{self.N}.txt"
        name_o = f"new_life\\res\\n_{self.N}\\range_zero_{self.N}.txt" 
        self.create_path(name_s[:17])
        self.create_path(name_n[:17])          
        with open(name_s,"w",encoding="utf-8") as file_s: 
            with open(name_n,"w",encoding="utf-8") as file_n:
                with open(name_o,"w",encoding="utf-8") as file_o: 
                    for i in self.sost:
                        for j in i:
                            tmp = self.eigenvalues(j)
                            lam = []
                            for k in range(len(tmp)):
                                lam.append(np.round(tmp[k],4))
                            z = 0
                            o = 0
                            for l in tmp.real:
                                if  l>-0.0005 and l<0.0005:
                                    o = 1
                                    break 
                                
                                if l < 0:
                                    z+=1
                            text = str(j)+'\t'+str(lam)+'\n' 
                            if z == 4 and o == 0:
                                file_s.write(text)
                            elif o==1:
                                file_o.write(text)
                            else:
                                file_n.write(text)
    
    #динамика для одной точки
    def dinamic(self,params = [2.094395, 4.18879, 1, 1, 2.0943951023931953]):
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.K,self.M,self.alpha = params 
        start_point[0] = start_point[0]+eps
        start_point[1] = start_point[1]+eps
        
        tmp = integrate.odeint(self.syst, start_point, self.t)
        plt.plot(self.t,tmp[:,0],label="x")
        plt.plot(self.t,tmp[:,1],label="y", linestyle = '--')
        plt.xlim(0, 100)
        plt.ylim(-10, 20)
        plt.legend()
        plt.show()
    
    #динамика, но сохраняем
    def rec_dinamic(self,way,z=1,params = [2.094395, 4.18879, 1, 1, 2.0943951023931953]):
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.K,self.M,self.alpha = params 
        start_point[0] += eps
        start_point[1] += eps
        # tmp = integrate.odeint(self.syst, start_point, self.t)
        tmp = solve_ivp(self.syst, [0,100], start_point, max_step = 0.1)
        plt.plot(tmp.t,tmp.y[0],label="x")
        plt.plot(tmp.t,tmp.y[1],label="y", linestyle = '--')
        # plt.plot(self.t,tmp[:,0],label="x")
        # plt.plot(self.t,tmp[:,1],label="y", linestyle = '--')
        plt.xlim(0, 100)
        plt.ylim(-10, 20)
        plt.legend()
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()

    #показываем графики, но выбираем какие и отдельно в папочки
    #ключевые слов "all", "st", "un_st"
    def show_sost(self,key = 'all'):
        n = self.N
        name = "new_life\\res\\n_\\"
        way = name
        
        if key == 'st':
            name = name+"stable_.txt"
            way = way+"stable\\"
        elif key == "un_st":
            name = name + "non_stable_.txt"
            way = way+"unstable\\"
        elif key == "rz":
            name = name + "range_zero_.txt"
            way = way+"range_zero\\"
        elif key == "all":
            name = name + "res_n_.txt"
            way = way+"all\\"
        else:
            print("wrong key")
            return
            
        sdvig1 = -4 
        sdvig2 = 15
        
        way_or = 'zamena\\'
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]
        # way = way[0:sdvig1]+f"{n}"+way[sdvig1:]
        way = way[0:sdvig2]+f"{n}"+way[sdvig2:] + way_or

        self.create_path(way)
        self.clean_path(way)
        # print(way)
        
        arr  = self.razbor_txt(name)
        
        rang = len(arr)
        if rang > MAX_GRAPH:
            rang = MAX_GRAPH
            
        for i in range(rang):
            self.rec_dinamic(params = arr[i],way = way,z=i+1)

    # чистим папку
    def clean_path(self,way):
        for filename in os.listdir(way):
            file_path = os.path.join(way, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        
    #разбор txt-шников в float массив
    def razbor_txt(self,name):
        ress = []
        with open(name) as file:
            for line in file:
                ress.append(self.razb_str(line.rstrip()))
        return ress
        
    def change_N(self,N_):
        self.N = N_

    def create_path(self, way):
        if not os.path.exists(way):
            os.makedirs(way)
            

    def razb_str(self, str):
        all = []
        tmp = ''

        for c in str:
            if c==' ' or c=='[':
                continue
            if c==',' or c==']':
                all.append(float(tmp))
                tmp = ''
                if c==']':
                    break
                continue
            tmp+=c
        return all

if __name__ == "__main__":
    tmp = [5 ,1]
    es = Equilibrium_states(p = tmp)
    # es.dinamic(params=[6.283185, 1.427449, 2, 1, 1.0471975511965976])
    # es.parall_st_eq() #подсчет всех состояний
    es.show_sost(key='st') #сохранение графиков #ключевые слов "all", "st", "un_st"
  