import numpy as np
# from requests import patch
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root
import joblib 
from numpy import linalg as LA
import os, shutil, sys
from sympy import Matrix, pretty
from scipy.integrate import solve_ivp
import ast

#Параметры системы 

col_razb = 10
MAX_GRAPH = 100
eps = 0
Max_time = 100


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
        self.t = np.linspace(0,Max_time,Max_time)

    # Сама система (возможно стоит использовать при расчете состояний равновесия)
    def syst(self,t,param):
        N = self.N
        M = self.M
        K = self.K
        omega = self.omega
        alpha = self.alpha
        m = self.m
        
        fi1,fi2,fi3,v,w,u = param #[x,y,w,v] - точки
        f = np.zeros(6)
        # fi1 with 1 dot
        f[0] = 1/m*(omega + 1/N * ((- K)*np.sin(alpha) + M*np.sin(fi2 - fi1 - alpha) + (N-M-K)*np.sin(fi3 - fi1 - alpha)) - v)
        # fi2 with 1 dot
        f[1] = 1/m*(omega + 1/N * (-M*np.sin(alpha) + K*np.sin(fi1 - fi2 - alpha) + (N-M-K)*np.sin(fi3 - fi2 - alpha)) - w)
        # fi3 with 1 dot
        f[2] = 1/m*(omega + 1/N * (-(N-M-K)*np.sin(alpha) + K*np.sin(fi1 - fi3 - alpha) + M*np.sin(fi2 - fi3 - alpha)) - u)
        # fi1 with 2 dots
        f[3] = v
        # fi2 with 2 dots
        f[4] = w
        # fi3 with 2 dots
        f[5] = u
        
        return f
    
    def full_syst(self,start_point):
        N = self.N
        omega = self.omega
        m = self.m
        M = self.M
        K = self.K
        alpha = self.alpha
        phi = np.zeros(N)
        v = np.zeros(N)
        for i in range(N):
            phi[i] = start_point[i]
            v[i] = start_point[i+N]
        f = np.zeros(2*N)

        f = np.zeros(2*N)
        
        for j in range(N):
            s = 0
            for phi_i in phi:
                s += np.sin(phi_i - phi[j] - alpha)
            
            f[j] = 1/N*s + omega - v[j]
            
            f[j+N] = v[j]
            
        return f
    #матрица Якоби
    def jakobi(self, param):

        fi1,fi2,fi3,M,K,alpha = param
        N = self.N
        m = self.m
        f = np.zeros(shape=(6,6))
        
        f[0]=[-1/m,
            0,
            0,
            1/m*(1/N * (-M*np.cos(fi2 - fi1 - alpha) - (N-M-K)*np.cos(fi3 - fi1 - alpha))),
            1/m*(1/N * (M*np.cos(fi2 - fi1 - alpha))),
            1/m*(1/N * ((N-M-K)*np.cos(fi3 - fi1 - alpha))),]
        f[1]=[0, 
            -1/m, 
            0,
            1/m*(1/N * (K*np.cos(fi1 - fi2 - alpha))),
            1/m*(1/N * (- K*np.cos(fi1 - fi2 - alpha) - (N-M-K)*np.cos(fi3 - fi2 - alpha))),
            1/m*(1/N * ((N-M-K)*np.cos(fi3 - fi2 - alpha)))]
        f[2]=[0, 
            0, 
            -1/m,
            1/m*(1/N * (K*np.cos(fi1 - fi3 - alpha))),
            1/m*(1/N * (M*np.cos(fi2 - fi3 - alpha))),
            1/m*(1/N * (-K*np.cos(fi1 - fi3 - alpha) - M*np.cos(fi2 - fi3 - alpha)))]
        f[3]=[1.0, 0, 0, 0, 0, 0]
        f[4]=[0, 1.0, 0, 0, 0, 0]
        f[5]=[0, 0, 1.0, 0, 0, 0]
        return f
    def jacobi_full(self,start_point):
        N = self.N
        m = self.m
        alpha = self.alpha
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
        print(res)
        return res
    
    def eigenvalues_full(self,start_point):
        matrix = self.jacobi_full(start_point)
        lam, vect = LA.eig(matrix)
        return lam
    
    def check_lams_full(self,params):
        N = self.N
        start_point = np.zeros(2*N)
        par = self.anti_zamena_2(arr=params)
        phi1,phi2,phi3,K,M,alpha = par
        self.alpha = alpha
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
        lam = self.eigenvalues_full(start_point)
        res_lam = []
        for l in lam:
            res_lam.append(np.round(complex(l),4))
        return res_lam

    def all_new_lam(self, key):
        way = f"new_life\\res\\n_{self.N}\\"
        if key == 'st':
            way += f"stable_{self.N}.txt"
        elif key == 'un_st':
            way += f"non_stable_{self.N}.txt"
        elif key == 'rz':
            way += f"range_zero_{self.N}.txt"
        res = []
        tmp=[]
        with open(way) as file:
            for line in file:
                tmp.append(self.razb_str(line.rstrip()))
        for x in tmp:
            res.append(self.check_lams_full(x))

        self.save_lamdas(res,key)
        return 0


    def save_lamdas(self, new_lam, key='st'):
        way_new = f"new_life\\res\\n_{self.N}\\new_lam.txt"
        if key == 'st':
            way_old = f"new_life\\res\\n_{self.N}\\stable_{self.N}.txt"
        elif key == 'un_st':
            way_old = f"new_life\\res\\n_{self.N}\\non_stable_{self.N}.txt"
        elif key == 'rz':
            way_old = f"new_life\\res\\n_{self.N}\\range_zero_{self.N}.txt"
        
        res = []
        with open(way_old,'r') as file:
            for line in file:
                res.append(self.razb_str_lam(line.rstrip()))
        old_lam = []
        for x in res:
            tmp = self.razb_str_2(x)
            old_lam.append(tmp)
        
        with open(way_new,"w",encoding="utf-8") as file:
            for i in range(len(new_lam)):
                file.write(str(new_lam[i]) + '\t' + str(old_lam[i]) + '\n')

        self.plot_lams(old_lam,new_lam,key)

    def razb_str_lam(self, str):
        res = []
        tmp = None
        count=0
        for c in str:
            if c=='[' or count!=2:
                if c=='[':
                    count+=1
                continue
            if c=='[' and count == 2:
                if len(tmp)!=0:
                    res.append(tmp)
                    tmp = ''
                continue
            if c==']':
                break
            if tmp is None:
                tmp=c
            else:
                tmp+=c
        if len(tmp)!=0:
            res.append(tmp)
        res = list(res)
        return res

    #якобиан
    def eigenvalues(self,param):
        matrix = self.jakobi(param)
        lam, vect = LA.eig(matrix)
        return lam
    
    def check_lams(self, params,i):
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
        elif count_st != 6 and count_rz==0:
            key = 'un_st'
        else:
            key = 'rz'
        # print("i: ", i, "Lamdas: ", lam,"\n", " Params: ", params, " key: ", key)
        return key

    #динамика для одной точки
    def dinamic(self, params = [np.pi, np.pi, 1, 1, np.pi/3]):
        par = self.anti_zamena(arr=params)
        par = np.reshape(par, (len(par[0])))
        start_point=np.zeros(6)
        start_point[0],start_point[1], start_point[2],self.M,self.K,self.alpha = par
        start_point[0] = start_point[0]+eps
        start_point[1] = start_point[1]+eps
        start_point[2] = start_point[2]+eps
        start_point[3] = eps
        start_point[4] = eps
        start_point[5] = eps
        
        # tmp = integrate.odeint(self.syst, start_point, self.t)
        res = solve_ivp(self.syst, [0,Max_time],start_point)
        plt.plot(res.t,np.sin(res.y[0]),label="sin(fi1)")
        plt.plot(res.t,np.sin(res.y[1]),label="sin(fi2)", linestyle = '--')
        plt.plot(res.t,np.sin(res.y[2]),label="sin(fi3)", linestyle = '-.')
        # plt.xlim(0, 100)
        plt.ylim(-1, 1)
        plt.text(x=Max_time//2,y=1.12, horizontalalignment = 'center', s="phi 1 = " + str(par[0]) + ", phi 2 = " + str(par[1]) + ", phi 3 = " + 
                str(par[2]) + ", K = " + str(par[3]) + ", M = " + str(par[4]) + ", alpha = " + str(par[5]))
        plt.legend()
        plt.show()
        
    
    #динамика, но сохраняем
    def rec_dinamic(self,way,z=1,params = [1, 2.094395, 4.18879, 1, 1, 2.0943951023931953]):
        start_point=np.zeros(6)
        start_point[0],start_point[1],start_point[2], self.M,self.K,self.alpha = params 
        start_point[0] += eps
        start_point[1] += eps
        start_point[2] += eps
        # tmp = integrate.odeint(self.syst, start_point, self.t)
        tmp = solve_ivp(self.syst, [0, Max_time],start_point, t_eval=self.t, trol=1e-11,atol=1e-11)
        # for x in tmp:
        #     x[0] = np.sin()
        plt.plot(tmp.t,np.sin(tmp.y[0]),label="sin(fi1)")
        plt.plot(tmp.t,np.sin(tmp.y[1]),label="sin(fi2)", linestyle = '--')
        plt.plot(tmp.t,np.sin(tmp.y[2]),label="sin(fi3)", linestyle = '-.')
        # plt.plot(tmp.t,tmp.y[3],label="fi1_with_dot")
        # plt.plot(tmp.t,tmp.y[4],label="fi2_with_dot", linestyle = '--')
        # plt.plot(tmp.t,tmp.y[5],label="fi3_with_dot", linestyle = '-.')
        # plt.xlim(0, 100)
        # plt.ylim(0, 1)
        plt.xlabel("t")
        plt.ylabel("sin(phi[i])")
        plt.text(x=Max_time//2,y=1.1, horizontalalignment = 'center', s="phi 1 = " + str(params[0]) + ", phi 2 = " + str(params[1]) + ", phi 3 = " + 
                str(params[2]) + ", K = " + str(params[3]) + ", M = " + str(params[4]) + ", alpha = " + str(params[5]))
        plt.legend()
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()
        return tmp

    def rec_dinamic_par(self, way, z, arr,t):
        R1 = self.order_parameter(arr)
        plt.plot(t, R1)
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
        
    def plot_lams(self, old_lams, new_lams, key):
        way = f"new_life\\res\\n_{self.N}\\"
        if key == 'st':
            way = way + "stable"
        elif key == 'un_st':
            way = way + "unstable"
        elif key == 'rz':
            way = way + "range_zero"
        elif key == 'all':
            way = way + "all"
        way += "\\lams"
        self.create_path(way)
        self.clean_path(way)
        way += "\\"
        for i in range(len(old_lams)):
            old_lam = np.array(old_lams[i])
            new_lam = np.array(new_lams[i])
            real_deal_old = old_lam.real
            not_real_deal_old = old_lam.imag
            real_deal_new = new_lam.real
            not_real_deal_new = new_lam.imag

            plt.scatter(real_deal_old, not_real_deal_old, c='b', marker='o')
            plt.scatter(real_deal_new, not_real_deal_new, c='r', marker='x')
            plt.grid()
            # plt.show()
            plt.savefig(way + f'graph_{i+1}.png')
            plt.clf()

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
            key = self.check_lams(arr[i],i)
            way_n = self.new_way(way, key)   
            # sdvig1 = -4 

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
            self.rec_dinamic_par(way = way_p,z=i+1, arr = tmp.y, t = tmp.t)
            # self.rec_dinamic_map(way=way_m, z=i+1, params=arr[i], res=res[i])
             
    def sost_in_fi(self, key = 'st'):
        n = self.N
        name = "new_life\\res\\n_\\"
        way = f"new_life\\res\\n_{self.N}\\"
        if key == 'st':
            name = name+"stable_.txt"
            way = way + "stable\\"
        elif key == "un_st":
            name = name + "non_stable_.txt"
            way = way + "unstable\\"
        elif key == "all":
            name = name + "res_n_.txt"
            way = way + "all\\"
        elif key == "rz":
            name = name + "range_zero_.txt"
            way = way + "range_zero\\"
        else:
            print("wrong key")
            return
        sdvig1 = -4 
        sdvig2 = 15
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]

        res = self.razbor_txt(name)
        res_fi = self.anti_zamena(res)
        
        way = way + "map\\"
        
        self.show_sost(arr = res_fi, key=key, res = res)
        ors.all_new_lam(key)
        tmp_count = 0
        for x in res:
            phi1 = self.up_arr(x,5,5)
            alpha = x[4]
            t_amx = 10000
            
            # phi1 = [2.474646, 2.474646, 2.474646, 0, 0]
            # alpha = 0
            
                
            # print(phi1)
            print(str(tmp_count+1)+":")
            self.plot_warm_map([phi1,eps,alpha,t_amx], way,tmp_count)
            tmp_count+=1

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
        ress = [fi1, fi2, fi3, int(arr[2]), int(arr[3]), arr[4]]
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

    def razb_str_2(self,str):
        all = []
        tmp = ''
        str=str[0]
        for c in str:
            if c==' ' or c=='[':
                continue
            if c==',' or c==']':
                all.append(complex(tmp))
                tmp = ''
                if c==']':
                    break
                continue

            tmp+=c
        all.append(complex(tmp))           
        return all

    def order_parameter(self, arr):
        res = []
        for i in range(Max_time):
            sumr = 0
            sumi = 0                
            for j in range(3):
                tmp = np.exp(arr[:,i][j]*1j)
                sumr += tmp.real
                sumi += tmp.imag
                sum = 1/3 * np.sqrt(sumr ** 2 + sumi ** 2)
            res.append(sum)
        # for x in arr:
            
        #     sumr = 0
        #     sumi = 0
        #     for i in range(3):
        #         tmp = np.exp(x[i]*1j)
        #         sumr += tmp.real
        #         sumi += tmp.imag
        #     sum = 1/3 * np.sqrt(sumr ** 2 + sumi ** 2)
        #     res.append(sum)
        return res
    
    #тепловая карта----------------------------------------------------------------------------------
    def iter_din(self, t, start_point , par):
    
        alpha,w = par
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
                s += np.sin(phi_i - phi[j] - alpha)
            
            # f[j] = round(s/lens + w - v[j], 7)
            # f[j+len(phi)] = round(v[j], 7)

            f[j]=s/lens + w - v[j]
            f[j+len(phi)] = v[j]

            s = 0
        # phi[:] = 0
        # v[:] = 0
        return f
        
    def din_thr_map(self, phi,v,par,t,t_max):
        start_point = np.zeros(len(phi)*2)
        for i in range(len(phi)):
            start_point[i] = phi[i]
            start_point[i+len(phi)] = v[i]

        res = solve_ivp(self.iter_din,[0,t_max],start_point, args=[par],rtol= 10e-10,atol=10e-10) # t_eval=t,
        
        return res.y

    def up_arr(self, arr,N,num_elems):
        res = np.array([])
        tmp = np.zeros(num_elems//N)
        
        if N>num_elems:
            num_elems = N
        
        razb = [int(arr[2]),int(arr[3]),int(N-arr[2]-arr[3])]
        
        for i in range(razb[0]):
            res = np.append(res,tmp)
        
        for i in range(len(razb[1:3])):
            tmp = tmp+arr[i]
            for j in range(razb[i+1]):
                res = np.append(res,tmp)
            tmp = tmp-arr[i]
        return res

    def plot_warm_map(self, param, way, count):
    
        phi,eps,alpha,t_max = param
        
        v = np.zeros(len(phi))

        for i in range(len(phi)):#
            phi[i] += eps
            v[i] += eps
        
        w = 1
        t = np.linspace(0,t_max,t_max)
        a = self.din_thr_map(phi,v,[alpha,w],t,t_max)
        
        matrix = np.array([])
        for i in range(len(phi)):
            matrix = np.append(matrix,a[i])
        
        matrix = matrix.reshape((len(phi),len(matrix)//len(phi)))

        l = len(phi) - 1
        while l>=0:
            # if l != 19:
            matrix[l] = matrix[l] - matrix[0]
            l-=1

        matrix = np.angle(np.exp(1j*matrix))
        print(matrix)
        plt.imshow(matrix, cmap ='hot',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect=4)

        plt.savefig(way + f'graph_{count+1}.png')
        plt.clf()


    #---------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    tmp = [5 , 1, 1]
    ors = Original_sist(p = tmp, fi = 0)
    # ors.dinamic(params=[[np.pi, 0.0, 1, 2, 2]])
    ors.sost_in_fi(key='un_st')