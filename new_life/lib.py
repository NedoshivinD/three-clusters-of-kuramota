import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root
import joblib 
from numpy import linalg as LA


#Параметры системы 

N_JOB = 4
col_razb = 10
MAX_GRAPH = 50
eps = 1

def razb_str(str):
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

class Tree_klasters(object):
    
    #инициализация системы
    def __init__(self,p = [3,1,1]):
        self.N,self.M,self.K = p
        self.alpha = 0
        self.m = 1
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,100,100)

    # Сама система (возможно стоит использовать при расчете состояний равновесия)
    def syst(self,param,t):
        N = self.N
        M = self.M
        K = self.K
        alpha = self.alpha
        m = self.m
        
        x,y,v,w = param #[x,y,w,v] - точки
        f = np.zeros(4)
        # x with 1 dot
        f[0] = 1/m*(1/N * ((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha)) - v)
        # y with 1 dot
        f[1] = 1/m*(1/N * ((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-(M)*np.sin(y-x-alpha)) - w)
        # x with 2 dots
        f[2] = v
        # y with 2 dots
        f[3] = w
        
        return f
    
    #поиск состояний равновесия для определенного параметра
    def state_equil(self,NMKA=[3,1,1,np.pi/2]):
        all_sol = [] 
        self.N,self.M,self.K,self.alpha = NMKA
        ran_x=[0,2*np.pi]
        ran_y=[0,2*np.pi]
        
        X = np.linspace(ran_x[0],ran_x[1],col_razb)
        Y = np.linspace(ran_y[0],ran_y[1],col_razb)
        v = 0
        w = 0
        
        for x in X:
            for y in Y:
                sol = root(self.syst,[x,y,v,w],args=(0),method='hybr')
                xar,yar,var,war = sol.x
                xar = round(xar,6)
                yar = round(yar,6)
                if [xar,yar,self.M,self.K,self.alpha] not in all_sol and (xar>=0 and xar<2*np.pi) and (yar<2*np.pi and yar>=0):   
                    all_sol.append([xar,yar,self.M,self.K,self.alpha])
                
        return all_sol
    
    #поиск всех состояний равновесия
    def parall_st_eq(self,N = 3):
        self.N = N
        M = np.linspace(1,N-2,N-2, dtype = 'int')
        K = np.linspace(N-2,1,N-2, dtype = 'int')
        
        alpha = np.linspace(0,np.pi,4)
        self.sost = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.state_equil)([N,m,k,al]) for m in M for k in K for al in alpha if m+k<N)
        
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
        x,y,M,K,alpha = param
        N = self.N
        m = self.m
        
        f = []
        f.append([0, 0, 1, 0])
        f.append([0, 0, 0, 1])
        f.append([1/(N*m)*(-M*np.cos(x+alpha) - K*np.cos(x - alpha) - (N - M - K)*np.cos(x-y-alpha)),
            1/(N*m)*(-(N-M-K)*np.cos(y + alpha) + (N-M-K)*np.cos(x-y-alpha)), -1/m, 0])
        f.append([1/(N*m)*(-M*np.cos(x+alpha) + M*np.cos(y-x-alpha)),
            1/(N*m)*(-K*np.cos(y-alpha)-(N-M-K)*np.cos(y+alpha)-M*np.cos(y-x-alpha)),
            0, -1/m])
        
        return(f)

    #поиск собственных чисел при определененных параметрах
    def eigenvalues(self,param):
        matrix = self.jakobi(param)
        lam, vect = LA.eig(matrix)
        return lam
    
    #Сохранение устойчивых и неустойчивых состояний равновесия 
    def rec_st_and_unst_st_eq(self):
        name_s = f"new_life\\res\\n_{self.N}\\stable_{self.N}.txt"
        name_n = f"new_life\\res\\n_{self.N}\\non_stable_{self.N}.txt"        
        with open(name_s,"w",encoding="utf-8") as file_s: 
            with open(name_n,"w",encoding="utf-8") as file_n: 
                for i in self.sost:
                    for j in i:
                        tmp = self.eigenvalues(j)
                        z = 0
                        for l in tmp.real:
                            if l < 0:
                                z+=1
                        text = str(j)+'\t'+str(tmp.real)+'\n' 
                        if z == 4:
                            file_s.write(text)
                        else:
                            file_n.write(text)
    
    #динамика для одной точки
    def dinamic(self,params = [2.094395, 4.18879, 1, 1, 2.0943951023931953]):
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.M,self.K,self.alpha = params 
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
        start_point[0],start_point[1], self.M,self.K,self.alpha = params 
        start_point[0] += eps
        start_point[1] += eps
        tmp = integrate.odeint(self.syst, start_point, self.t)
        plt.plot(self.t,tmp[:,0],label="x")
        plt.plot(self.t,tmp[:,1],label="y", linestyle = '--')
        plt.xlim(0, 100)
        plt.ylim(-10, 20)
        plt.legend()
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()

    #показываем графики, но выбираем какие и отдельно в папочки
    #ключевые слов "all", "st", "non_st"
    def show_sost(self,key = 'all',n=3):
        name = "new_life\\res\\n_\\"
        way = name
        
        if key == 'st':
            name = name+"stable_.txt"
            way = way+"stable\\"
        elif key == "un_st":
            name = name + "non_stable_.txt"
            way = way+"unstable\\"
        else :
            name = name + "res_n_.txt"
            way = way+"all\\"
            
        sdvig1 = -4 
        sdvig2 = 15
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]
        # way = way[0:sdvig1]+f"{n}"+way[sdvig1:]
        way = way[0:sdvig2]+f"{n}"+way[sdvig2:]
        
        # print(way)
        
        arr  = self.razbor_txt(name)
        
        rang = len(arr)
        if rang > MAX_GRAPH:
            rang = MAX_GRAPH
            
        for i in range(rang):
            self.rec_dinamic(params = arr[i],way = way,z=i+1)

        
    #разбор txt-шников в float массив
    def razbor_txt(self,name):
        ress = []
        with open(name) as file:
            for line in file:
                ress.append(razb_str(line.rstrip()))
        return ress
        
    def change_N(self,N_):
        self.N = N_



if __name__ == "__main__":
    tmp = [3,1,1]
    tk = Tree_klasters(p = tmp)
    
    tk.dinamic(params=[3.141593, 3.141593, 1, 1, 3.141592653589793])
    # tk.parall_st_eq() #подсчет всех состояний
    # tk.show_sost(key='st') #сохранение графиков

#govno v 20,30