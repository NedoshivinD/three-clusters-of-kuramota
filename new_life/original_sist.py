import numpy as np
from requests import patch
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root
import joblib 
from numpy import linalg as LA
import os, shutil


#Параметры системы 

col_razb = 10
MAX_GRAPH = 50
eps = 0.3


class Original_sist(object):
    
    #инициализация системы
    def __init__(self,p = [3,1, 1], fi=1):
        self.N, self.m,self.omega = p
        self.M = 0
        self.K = 0
        self.alpha = 0
        self.fi1 = fi
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,100,100)

    # Сама система (возможно стоит использовать при расчете состояний равновесия)
    def syst(self,param,t):
        N = self.N
        M = self.M
        K = self.K
        omega = self.omega
        alpha = self.alpha
        m = self.m
        
        fi1,fi2,fi3,v,w,u = param #[x,y,w,v] - точки
        f = np.zeros(6)
        # fi1 with 1 dot
        f[0] = 1/m*(omega + 1/N * ((- K)*np.sin(alpha) + M*np.sin(fi2 - fi1 + alpha) + (N-M-K)*np.sin(fi3 - fi1 + alpha)) - v)
        # fi2 with 1 dot
        f[1] = 1/m*(omega + 1/N * (-M*np.sin(alpha) + K*np.sin(fi1 - fi2 + alpha) + (N-M-K)*np.sin(fi3 - fi2 + alpha)) - w)
        # fi3 with 1 dot
        f[2] = 1/m*(omega + 1/N * (-(N-M-K)*np.sin(alpha) + K*np.sin(fi1 - fi3 + alpha) + M*np.sin(fi2 - fi3 + alpha)) - u)
        # fi1 with 2 dots
        f[3] = v
        # fi2 with 2 dots
        f[4] = w
        # fi3 with 2 dots
        f[5] = u
        
        return f
    
    #динамика для одной точки
    def dinamic(self,params = [np.pi, np.pi, 1, 1, np.pi/3]):
        tmp = self.anti_zamena(arr=params)
        start_point=np.zeros(6)
        start_point[0],start_point[1], start_point[2],self.M,self.K,self.alpha = tmp[0] 
        start_point[0] = start_point[0]+eps
        start_point[1] = start_point[1]+eps
        start_point[2] = start_point[2]+eps
        
        tmp = integrate.odeint(self.syst, start_point, self.t)
        plt.plot(self.t,tmp[:,0] - tmp[:,0],label="fi1")
        plt.plot(self.t,tmp[:,1] - tmp[:,0],label="fi2", linestyle = '--')
        plt.plot(self.t,tmp[:,2] - tmp[:,0],label="fi3", linestyle = '-.')
        # plt.xlim(0, 100)
        # plt.ylim(0, 100)
        plt.legend()
        plt.show()
    
    #динамика, но сохраняем
    def rec_dinamic(self,way,z=1,params = [1, 2.094395, 4.18879, 1, 1, 2.0943951023931953]):
        start_point=np.zeros(6)
        start_point[0],start_point[1],start_point[2], self.M,self.K,self.alpha = params 
        start_point[0] += eps
        start_point[1] += eps
        start_point[2] += eps
        tmp = integrate.odeint(self.syst, start_point, self.t)
        # for x in tmp:
        #     x[0] = np.sin()
        plt.plot(self.t,tmp[:,0] - tmp[:,0],label="fi1")
        plt.plot(self.t,tmp[:,1] - tmp[:,0],label="fi2", linestyle = '--')
        plt.plot(self.t,tmp[:,2] - tmp[:,0],label="fi3", linestyle = '-.')
        # plt.xlim(0, 100)
        # plt.ylim(0, 1)
        plt.legend()
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()

    #показываем графики, но выбираем какие и отдельно в папочки
    #ключевые слов "all", "st", "un_st"
    def show_sost(self,arr, key = 'all'):
        n = self.N
        name = "new_life\\res\\n_\\"
        way = name
        
        if key == 'st':
            way = way+"stable\\"
        elif key == "un_st":
            way = way+"unstable\\"
        elif key == "r_o":
            way = way+"range_zero\\"
        elif key == "all":
            way = way+"all\\"
        else:
            print("wrong key")
            return
            
        # sdvig1 = -4 
        sdvig2 = 15
        way_or = 'origin\\'
        

        # way = way[0:sdvig1]+f"{n}"+way[sdvig1:]
        way = way[0:sdvig2]+f"{n}"+way[sdvig2:] + way_or

        self.create_path(way)
        self.clean_path(way)
        # print(way)
                
        rang = len(arr)
        if rang > MAX_GRAPH:
            rang = MAX_GRAPH
            
        for i in range(rang):
            self.rec_dinamic(params = arr[i],way = way,z=i+1)

    def sost_in_fi(self, key = 'all'):
        n = self.N
        name = "new_life\\res\\n_\\"
        
        if key == 'st':
            name = name+"stable_.txt"
        elif key == "un_st":
            name = name + "non_stable_.txt"
        elif key == "all":
            name = name + "res_n_.txt"
        else:
            print("wrong key")
            return
        sdvig1 = -4 
        sdvig2 = 15
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]
        # print(name)
        res = self.razbor_txt(name)
        res_fi = self.anti_zamena(res)
        self.show_sost(arr = res_fi, key=key)


    def anti_zamena(self, arr):
        ress = []
        fi1 = self.fi1
        for x in arr:
            fi2 = fi1 - x[0]
            fi3 = fi1 - x[1]
            ress.append([fi1, fi2, fi3, x[2], x[3], x[4]])
        return ress
                
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
    def razb_str(self,str):
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
    tmp = [4,1, 1]
    ors = Original_sist(p = tmp)
    ors.dinamic(params=[[6.283185, 1.427449, 2, 1, 1.0471975511965976]])
    # ors.sost_in_fi(key='st')
    


    # np.angel(fin - fi0)
    # параметр порядка

    # посмотреть 2х кластерное разбиение но со второй гармоникой