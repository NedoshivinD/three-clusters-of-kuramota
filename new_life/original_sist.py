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
MAX_GRAPH = 100
eps = 0.1


class Original_sist(object):
    
    #инициализация системы
    def __init__(self,p = [3,1, 1], fi=1):
        self.N, self.m,self.omega = p
        self.M = 0
        self.K = 0
        self.alpha = 0
        self.fi1 = fi
        self.N_fi1 = 10
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
    
    #матрица Якоби
    def jakobi(self, param):
        tmp = self.anti_zamena_2(arr=param)

        fi1,fi2,fi3,M,K,alpha = tmp
        N = self.N
        m = self.m
        f = np.zeros(shape=(6,6))
        f[0]=[0, 0, 0, 1.0, 0, 0]
        f[1]=[0, 0, 0, 0, 1.0, 0]
        f[2]=[0, 0, 0, 0, 0, 1.0]
        f[3]=[1/m*(1/N * (-M*np.cos(fi2 - fi1 + alpha) - (N-M-K)*np.cos(fi3 - fi1 + alpha))),
            1/m*(1/N * (M*np.cos(fi2 - fi1 + alpha))),
            1/m*(1/N * ((N-M-K)*np.cos(fi3 - fi1 + alpha))),
            -1/m,
            0,
            0]
        f[4]=[1/m*(1/N * (K*np.cos(fi1 - fi2 + alpha))),
            1/m*(1/N * (- K*np.cos(fi1 - fi2 + alpha) - (N-M-K)*np.cos(fi3 - fi2 + alpha))),
            1/m*(1/N * ((N-M-K)*np.cos(fi3 - fi2 + alpha))),
            0, 
            -1/m, 
            0]
        f[5]=[1/m*(1/N * (K*np.cos(fi1 - fi3 + alpha))),
            1/m*(1/N * (M*np.cos(fi2 - fi3 + alpha))),
            1/m*( 1/N * (-K*np.cos(fi1 - fi3 + alpha) - M*np.cos(fi2 - fi3 + alpha))),
            0, 
            0, 
            -1/m]
        return f
    #якобиан
    def eigenvalues(self,param):
        matrix = self.jakobi(param)
        lam, vect = LA.eig(matrix)
        return lam
    
    def check_lams(self, params):
        lam = self.eigenvalues(params)
        count_st = 0
        count_unst = 0
        count_rz = 0
        for l in lam.real:
            if l < 0:
                count_st+=1
            if l > 0:
                count_unst+=1
            if l == 0:
                count_rz+=1
        if count_st == 6:
            key = 'st'
        if count_st != 6 and count_rz==0:
            key = 'un_st'
        if count_rz!=0 and count_st == 0 and count_unst == 0:
            key = 'rz'
        # print("Lamdas: ", lam, " Params: ", params, " key: ", key)
        return key

    #динамика для одной точки
    def dinamic(self, params = [np.pi, np.pi, 1, 1, np.pi/3]):
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
        return tmp

    def rec_dinamic_par(self, way, z, arr):
        R1 = self.order_parameter(arr)
        plt.plot(self.t, R1)
        # plt.xlim(0, 100)
        plt.ylim(0, 1.1)
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()

    def rec_dinamic_map(self, way, z, params, res):
        start_point=np.zeros(6)
        start_point[0],start_point[1],start_point[2], self.M,self.K,self.alpha = params
        x = res[0]
        y = res[1] 
        start_point[0] += eps
        start_point[1] += eps
        start_point[2] += eps
        fi1_arr = np.linspace(0, 2*np.pi, self.N_fi1)
        for i in range(self.N_fi1):
            start_point[0] = fi1_arr[i]
            new_points = self.anti_zamena_2([start_point[0],x,y, start_point[3],start_point[4],start_point[5]])
            tmp = integrate.odeint(self.syst, new_points, self.t)
            # c = tmp[:,0] - tmp[:,1]# - tmp[:,0]
            plt.plot(self.t,tmp[:,0],label="fi1", c='r', alpha = 0.5)
            plt.plot(self.t,tmp[:,1],label="fi2", c='b', alpha = 0.5)
            plt.plot(self.t,tmp[:,2],label="fi3", c='g', alpha = 0.5)
            # plt.xlim(0, 100)
            # plt.ylim(0, 1)
        plt.show()
        # plt.legend()
        # if way == 'new_life\\res\\n_4\\unstable\\map\\' and z == 2:
        #     plt.show()

        # print("Way: ", way, " z: ", z)
        # plt.savefig(way + f'graph_{z}.png')
        # plt.clf()
        
    def new_way(self, way, key):
        if key == 'st':
            way = way+"stable\\"
        elif key == "un_st":
            way = way+"unstable\\"
        elif key == "rz":
            way = way+"range_zero\\"
        elif key == "all":
            way = way+"all\\"
        else:
            print("wrong key")
            return
        return way

    #показываем графики, но выбираем какие и отдельно в папочки
    #ключевые слов "all", "st", "un_st"
    def show_sost(self,arr, key, res):
        n = self.N
        name = "new_life\\res\\n_\\"
        way = name
        rang = len(arr)
        if rang > MAX_GRAPH:
            rang = MAX_GRAPH
        
        way1 = way+"stable\\"
        way2 = way+"unstable\\"
        way3 = way+"range_zero\\"
        way4 = way+"all\\"
        sdvig2 = 15
        way_or = 'origin\\'
        way_par = 'order_params\\'
        way_map = 'map\\'

        way_m1 = way1[0:sdvig2]+f"{n}"+way1[sdvig2:] + way_map
        way_p1 = way1[0:sdvig2]+f"{n}"+way1[sdvig2:] + way_par
        way1 = way1[0:sdvig2]+f"{n}"+way1[sdvig2:] + way_or
        self.create_path(way1)
        self.clean_path(way1)
        self.create_path(way_p1)
        self.clean_path(way_p1)
        self.create_path(way_m1)
        self.clean_path(way_m1)

        way_m2 = way2[0:sdvig2]+f"{n}"+way2[sdvig2:] + way_map
        way_p2 = way2[0:sdvig2]+f"{n}"+way2[sdvig2:] + way_par
        way2 = way2[0:sdvig2]+f"{n}"+way2[sdvig2:] + way_or
        self.create_path(way2)
        self.clean_path(way2)
        self.create_path(way_p2)
        self.clean_path(way_p2)
        self.create_path(way_m2)
        self.clean_path(way_m2)

        way_m3 = way3[0:sdvig2]+f"{n}"+way3[sdvig2:] + way_map
        way_p3 = way3[0:sdvig2]+f"{n}"+way3[sdvig2:] + way_par
        way3 = way3[0:sdvig2]+f"{n}"+way3[sdvig2:] + way_or
        self.create_path(way3)
        self.clean_path(way3)
        self.create_path(way_p3)
        self.clean_path(way_p3)
        self.create_path(way_m3)
        self.clean_path(way_m3)

        way_m4 = way4[0:sdvig2]+f"{n}"+way4[sdvig2:] + way_map
        way_p4 = way4[0:sdvig2]+f"{n}"+way4[sdvig2:] + way_par
        way4 = way4[0:sdvig2]+f"{n}"+way4[sdvig2:] + way_or
        self.create_path(way4)
        self.clean_path(way4)
        self.create_path(way_p4)
        self.clean_path(way_p4)
        self.create_path(way_m4)
        self.clean_path(way_m4)

        for i in range(rang):
            key = self.check_lams(arr[i])
            way_n = self.new_way(way, key)   
            # sdvig1 = -4 
            sdvig2 = 15
            way_or = 'origin\\'
            way_par = 'order_params\\'
            way_map = 'map\\'

            # way = way[0:sdvig1]+f"{n}"+way[sdvig1:]
            way_m = way_n[0:sdvig2]+f"{n}" + way_n[sdvig2:] + way_map
            way_p = way_n[0:sdvig2]+f"{n}" + way_n[sdvig2:] + way_par
            way_k = way_n[0:sdvig2]+f"{n}" + way_n[sdvig2:] + way_or
            
            # print("Way: ", way_k, " I: ", i) 
        #     self.create_path(way) 
        #     self.clean_path(way)
        #     self.create_path(way_p)
        #     self.clean_path(way_p)
        #     self.create_path(way_m)
        #     self.clean_path(way_m)
        # # print(way)
                
        # rang = len(arr)
        # if rang > MAX_GRAPH:
        #     rang = MAX_GRAPH
            
        # for i in range(rang):
            tmp = self.rec_dinamic(params = arr[i],way = way_k,z=i+1)
            self.rec_dinamic_par(way = way_p,z=i+1, arr = tmp)
            # self.rec_dinamic_map(way=way_m, z=i+1, params=arr[i], res=res[i])
             
    def sost_in_fi(self, key = 'all'):
        n = self.N
        name = "new_life\\res\\n_\\"
        
        if key == 'st':
            name = name+"stable_.txt"
        elif key == "un_st":
            name = name + "non_stable_.txt"
        elif key == "all":
            name = name + "res_n_.txt"
        elif key == "rz":
            name = name + "range_zero_.txt"
        else:
            print("wrong key")
            return
        sdvig1 = -4 
        sdvig2 = 15
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]

        res = self.razbor_txt(name)
        res_fi = self.anti_zamena(res)


        self.show_sost(arr = res_fi, key=key, res = res)
        # ress = self.order_parameter(res_fi)
        # self.show_sost(arr = ress, key=key)

    # просто рисовалка) 
    # def PlotOnPlane(self, arr):
    #     plt.plot()
    def anti_zamena(self, arr):
        ress = []
        fi1 = self.fi1
        for x in arr:
            fi2 = fi1 - x[0]
            fi3 = fi1 - x[1]
            ress.append([fi1, fi2, fi3, x[2], x[3], x[4]])
        return ress
    
    def anti_zamena_2(self, arr):
        fi1 = self.fi1
        fi2 = fi1 - arr[0]
        fi3 = fi1 - arr[1]
        ress = [fi1, fi2, fi3, arr[2], arr[3], arr[4]]
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

    def order_parameter(self, arr):
        res = []
        for x in arr:
            sumr = 0
            sumi = 0
            for i in range(3):
                tmp = np.exp(x[i]*1j)
                sumr += tmp.real
                sumi += tmp.imag
            sum = 1/3 * np.sqrt(sumr ** 2 + sumi ** 2)
            res.append(sum)
        return res

if __name__ == "__main__":
    tmp = [4,np.pi, 0]
    ors = Original_sist(p = tmp, fi = 1)
    # ors.dinamic(params=[[6.283185, 1.427449, 2, 1, 1.0471975511965976]])
    ors.sost_in_fi(key='all')
    


    # np.angel(fin - fi0)
    # параметр порядка

    # посмотреть 2х кластерное разбиение но со второй гармоникой