import numpy as np
from requests import patch
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root
import joblib 
from numpy import linalg as LA
import os, shutil
from scipy.integrate import solve_ivp
from supp_func import razb_config
from matplotlib.backends.backend_pdf import PdfPages
import time

#Параметры системы 


class Original_sist(object):
    
    #инициализация системы
    def __init__(self,p = [3,1,0], fi=1):
        self.N, self.m,self.omega = p
        self.M = 0
        self.k1 = 1
        self.k2 = 1
        self.alpha = 0
        self.beta = 0
        self.fi1 = fi
        self.N_fi1 = 10
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,100,100)
        self.rand_one = 0
        self.rand_two = 0
        self.rand_koleb = 0

    # Сама система (возможно стоит использовать при расчете состояний равновесия)
    def syst_orig(self,t,param):
        N = self.N
        M = self.M
        omega = self.omega
        alpha = self.alpha
        beta = self.beta
        m = self.m
        k1 = self.k1
        k2 = self.k2
        
        
        fi1,fi2,v,w = param #[x,y,w,v] - точки
        f = np.zeros(4)

        # fi1 with 1 dots
        f[0] = v
        # fi2 with 1 dots
        f[1] = w
        # fi1 with 2 dot
        f[2] = 1/m*( 1/N * ( (-M) *(k1*np.sin(alpha) +k2*np.sin(beta)) + (N-M)*(k1*np.sin(fi2 - fi1 - alpha) + k2*np.sin(2*fi2 - 2*fi1 - beta)))  -v + omega)
        # fi2 with 2 dot
        f[3] = 1/m*( 1/N * ( -(N-M) *(k1*np.sin(alpha) +k2*np.sin(beta)) + M*(k1*np.sin(fi1 - fi2 - alpha) + k2*np.sin(2*fi1 - 2*fi2 - beta)))  - w + omega)
        
        return f
    
    def full_syst(self,start_point):
        N = self.N
        omega = self.omega
        m = self.m
        M = self.M
        alpha = self.alpha
        beta = self.beta
        phi = np.zeros(N)
        v = np.zeros(N)
        k1 = self.k1
        k2 = self.k2

        for i in range(N):
            phi[i] = start_point[i]
            v[i] = start_point[i+N]
        f = np.zeros(2*N)
        
        for j in range(N):
            s = 0
            for phi_i in phi:
                s += np.sin(phi_i - phi[j] - alpha) + np.sin(2*(phi_i - phi[j]) - beta)
            
            f[j] = 1/m * (1/N*s + omega - v[j])
            
            f[j+N] = v[j]
            
        return f
    
    #матрица Якоби
    def jakobi(self, param):

        fi1,fi2,M,alpha,beta,k1,k2 = param
        N = self.N
        m = self.m
        f = np.zeros(shape=(4,4))
        
        f[0]=[-1/m,
            0,
            1/m*(1/N * (N -M) * (-k1*np.cos(fi2-fi1-alpha) - 2*k2*np.cos(2*fi2-2*fi1-beta))),
            1/m*(1/N * (N -M) * ( k1*np.cos(fi2-fi1-alpha) + 2*k2*np.cos(2*fi2-2*fi1-beta))),]
        f[1]=[0, 
            -1/m,
            1/m*(1/N * (M)    * ( k1*np.cos(fi1-fi2-alpha) + 2*k2*np.cos(2*fi1-2*fi2-beta))),
            1/m*(1/N * (M)    * (-k1*np.cos(fi1-fi2-alpha) - 2*k2*np.cos(2*fi1-2*fi2-beta)))]
        f[2]=[1.0, 0, 0, 0]
        f[3]=[0, 1.0, 0, 0]
        return f
    
    def jacobi_full(self,start_point):#!
        N = self.N
        m = self.m
        alpha = self.alpha
        beta = self.beta
        k1 = self.k1
        k2 = self.k2
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
                            sum+=k1*np.cos(phi[j]-phi[k]+alpha) + k2*2*np.cos(2*(phi[j]-phi[k])+beta)
                    tmp_arr[j] = - 1/(m*N) * sum
                else:
                    tmp_arr[j] = 1/(m*N) * (k1*np.cos(phi[j]-phi[i]+alpha) + k2*2*np.cos(2*(phi[j]-phi[i])+beta))

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
    
    def eigenvalues_full(self,start_point):
        matrix = self.jacobi_full(start_point)
        lam, vect = LA.eig(matrix)
        return lam
    
    def check_lams_full(self,params):
        N = self.N
        start_point = np.zeros(2*N)
        par = self.anti_zamena_2(arr=params)
        phi1,phi2,M,alpha,beta,k1,k2 = par
        self.alpha = alpha
        self.beta = beta
        self.k1 = k1
        self.k2 = k2
        v1,v2 = (0,0)
        for i in range(M):
            start_point[i] = phi1
            start_point[i+N] = v1
        for i in range(N-M):
            start_point[M+i] = phi2
            start_point[M+i+N] = v2
        lam = self.eigenvalues_full(start_point)
        res_lam = []
        for l in lam:
            res_lam.append(np.round(complex(l),4))
        return res_lam

    def all_new_lam(self, key):
        way = f"2_garmonic\\res\\n_{self.N}\\"
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
        way_new = f"2_garmonic\\res\\n_{self.N}\\new_lam.txt"
        if key == 'st':
            way_old = f"2_garmonic\\res\\n_{self.N}\\stable_{self.N}.txt"
        elif key == 'un_st':
            way_old = f"2_garmonic\\res\\n_{self.N}\\non_stable_{self.N}.txt"
        elif key == 'rz':
            way_old = f"2_garmonic\\res\\n_{self.N}\\range_zero_{self.N}.txt"
        
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
        if count_st == len(lam):
            key = 'st'
        elif count_st != len(lam) and count_rz==0:
            key = 'un_st'
        else:
            key = 'rz'
        # print("i: ", i, "Lamdas: ", lam,"\n", " Params: ", params, " key: ", key)
        return key


    #динамика для одной точки
    def dinamic_klast(self, params = [np.pi, 1, np.pi/3, np.pi/3]):
        par = self.anti_zamena(arr=[params])
        par = np.reshape(par, (len(par[0])))
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.M,self.alpha, self.beta,self.k1,self.k2 = par 
        start_point[0] = start_point[0]+eps
        start_point[1] = start_point[1]+eps
        start_point[2] = eps
        start_point[3] = eps
        
        t = np.linspace(0,Max_time,1000)
        # tmp = integrate.odeint(self.syst, start_point, self.t)
        res = solve_ivp(self.syst_orig, [0,Max_time], start_point, t_eval=t)
        plt.plot(res.t,np.sin(res.y[0]),label="sin(fi1)")
        plt.plot(res.t,np.sin(res.y[1]),label="sin(fi2)", linestyle = '--')
        # plt.xlim(0, 100)
        # plt.ylim(0, 100)
        plt.ylim(-1, 1)
        plt.xlabel("t")
        plt.ylabel(r"$sin(\phi_i)$")

        plt.legend()
        plt.show()
    
    #динамика, но сохраняем
    def rec_dinamic(self,way,z=1,par = [3.141593, 1, 0.0, 0.0, 1, 1]):
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.M,self.alpha, self.beta,self.k1,self.k2 = par 
        start_point[0] += eps
        start_point[1] += eps
        t = np.linspace(0,100,1000)
        # tmp = integrate.odeint(self.syst, start_point, self.t)
        tmp = solve_ivp(self.syst_orig, [0, Max_time],start_point, t_eval=t, trol=1e-11,atol=1e-11)
        # for x in tmp:
        #     x[0] = np.sin()
        plt.plot(tmp.t,np.sin(tmp.y[0]),label=r"$sin(\phi_1)$")
        plt.plot(tmp.t,np.sin(tmp.y[1]),label=r"$sin(\phi_2)$", linestyle = '--')
        # plt.xlim(0, 100)
        # plt.ylim(0, 1)
        plt.xlabel("t")
        plt.ylabel(r"$sin(\phi_i)$")
        # plt.text(x=Max_time//2,y=1.12, horizontalalignment = 'center', s="phi 1 = " + str(par[0]) + ", phi 2 = " + str(par[1]) +
        #          ", M = " + str(par[2]) + ", alpha = " + str(par[3]) + ", beta = " + str(par[4])+ ", k1 = " + str(par[5])+ ", k2 = " + str(par[6]))
        plt.legend()
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()
        return tmp

    def rec_dinamic_par(self, way, z, arr,t):
        R1 = self.order_parameter(arr)
        plt.plot(t, R1)
        # plt.xlim(0, 100)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("t")
        plt.ylabel('z')
        plt.title('order parametr')
        plt.savefig(way + f'graph_{z}.png')
        plt.clf()


    def rec_dinamic_map(self, way, z, params, res):
        start_point=np.zeros(4)
        start_point[0],start_point[1], self.M, self.alpha, self.beta, self.k1,self.k2 = params
        x = res[0]
        start_point[0] += eps
        start_point[1] += eps
        fi1_arr = np.linspace(0, 2*np.pi, self.N_fi1)
        t = np.linspace(0,100,1000)
        for i in range(self.N_fi1):
            start_point[0] = fi1_arr[i]
            new_points = self.anti_zamena_2([start_point[0],x, start_point[2],start_point[3],start_point[4],start_point[5],start_point[6]])
            tmp = integrate.odeint(self.syst_orig, new_points, t)
            # c = tmp[:,0] - tmp[:,1]# - tmp[:,0]
            plt.plot(self.t,tmp[:,0],label="fi1", c='r', alpha = 0.5)
            plt.plot(self.t,tmp[:,1],label="fi2", c='b', alpha = 0.5)
            # plt.xlim(0, 100)
            # plt.ylim(0, 1)
        plt.show()

    def plot_lams(self, old_lams, new_lams, key):
        way = f"2_garmonic\\res\\n_{self.N}\\"
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

            plt.scatter(real_deal_old, not_real_deal_old, c='b', marker='o',label = r'$\lambda$ reduced')
            plt.scatter(real_deal_new, not_real_deal_new, c='r', marker='x',label = r'$\lambda$ original')
            plt.grid()
            plt.legend()
            plt.xlabel(r'Re($\lambda$)')
            plt.ylabel(r'Im($\lambda$)')
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
        name = f"2_garmonic\\res\\n_{self.N}\\"
        way = name
        rang = len(arr)
        if rang > MAX_GRAPH:
            rang = MAX_GRAPH

        if key == 'st':
            way_all = way + "stable\\"
        elif key == 'un_st':
            way_all = way+"unstable\\"
        elif key == 'rz':
            way_all = way+"range_zero\\"
        elif key == 'all':
            way_all = way+"all\\"
        
        way_par = way_all + 'order_params\\'
        way_map = way_all + 'map\\'
        way_orig = way_all + 'origin\\'

        self.create_path(way_orig)
        self.clean_path(way_orig)

        self.create_path(way_par)
        self.clean_path(way_par)
        
        self.create_path(way_map + 'png')
        self.clean_path(way_map + 'png')
        
        self.create_path(way_map + 'pdf')
        self.clean_path(way_map + 'pdf')
        
        self.create_path(way_map + 'svg')
        self.clean_path(way_map + 'svg')

        for i in range(len(arr)):
            key = self.check_lams(arr[i],i)
            
            way_p = way_all + 'order_params\\'
            way_k = way_all + 'origin\\'
                 
            tmp = self.rec_dinamic(par = arr[i],way = way_k,z=i+1)
            self.rec_dinamic_par(way = way_p,z=i+1, arr = tmp.y, t = tmp.t)

    
    def sost_in_fi(self, key = 'all'):
        n = self.N
        name = "2_garmonic\\res\\n_\\"
        way = f"2_garmonic\\res\\n_{self.N}\\"
        
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
        sdvig2 = 17
        
        name = name[0:sdvig1]+f"{n}"+name[sdvig1:]
        name = name[0:sdvig2]+f"{n}"+name[sdvig2:]

        res = self.razbor_txt(name)
        res_fi = self.anti_zamena(res)


        self.show_sost(arr = res_fi, key=key, res =res)
        ors.all_new_lam(key)
        tmp_count = 0
        start_phi = START_PHI

        way = way + "map\\"
        
        if HOT:
            joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.__work_hot_plot__)(x,res,start_phi,way) for x in res)

    
    def __work_hot_plot__(self,x,res,start_phi,way,dop_title = ''): 
        phi1 = self.up_arr(start_phi,x,self.N,self.N)
        alpha = x[2]
        beta = x[3]
        k1 = x[4]
        k2 = x[5]
        t_amx = 10000

        tmp_count = res.index(x)
        if dop_title != '':
            if tmp_count  == 0:
                dop_title += 'koleb'
            if tmp_count  == 1:
                dop_title += 'one'
            if tmp_count  == 2:
                dop_title += 'two'
        
        self.plot_warm_map([phi1,eps,alpha,beta,k1,k2,t_amx], way,tmp_count, dop_title)

        
    def anti_zamena(self, arr):
        ress = []
        fi1 = self.fi1
        for x in arr:
            fi2 = fi1 - x[0]
            ress.append([fi1, fi2, x[1], x[2], x[3], x[4], x[5]])
        return ress

    def anti_zamena_2(self, arr):
        fi1 = self.fi1
        fi2 = fi1 - arr[0]
        ress = [fi1, fi2, int(arr[1]), arr[2], arr[3], arr[4], arr[5]]
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
        for i in range(len(arr[0])):
            sumr = 0
            sumi = 0                
            for j in range(2):
                tmp = np.exp(arr[:,i][j]*1j)
                sumr += tmp.real
                sumi += tmp.imag
                sum = 1/2 * np.sqrt(sumr ** 2 + sumi ** 2)
            res.append(sum)
        return res
    
    #тепловая карта----------------------------------------------------------------------------------
    def iter_din(self, t, start_point , par):
        
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

    def din_thr_map(self,phi,v,par,t,t_max):
        start_point = np.zeros(len(phi)*2)
        for i in range(len(phi)):
            start_point[i] = phi[i]
            start_point[i+len(phi)] = v[i]

        res = solve_ivp(self.iter_din,[0,t_max],start_point, args=[par],rtol= 10e-10,atol=10e-10) # t_eval=t,
        
        return res.y

    def up_arr(self,start_phi,arr,N,num_elems):
        res = np.array ([])
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
        
        for i in range(int(razb[0])):
            res = np.append(res,tmp+eps_map)
        
        tmp-= start_phi
        
        tmp = tmp+arr[0]
        
        for i in range(int(razb[1])):
            res = np.append(res, start_phi - tmp)
            
        tmp = tmp-arr[0]
        
        return res

    def plot_warm_map(self, param, way, count,dop_title = ''):
    
        phi,eps,alpha,beta,k1,k2,t_max = param
        
        v = np.zeros(len(phi))
        v = v + 1e-3

        w = 1
        t = np.linspace(0,t_max,t_max)
        a = self.din_thr_map(phi,v,[alpha,beta,k1,k2,w],t,t_max)
        
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
        pdf = PdfPages(way +'pdf\\'+ f'graph_{count+1}.pdf')
        
        matrix = np.angle(np.exp(1j*matrix))
        print(str(count+1)+":")
        print(matrix)
        # plt.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)*10], aspect=4)
        fig, ax = plt.subplots()
        p = ax.imshow(matrix, cmap ='hsv',vmin=-np.pi, vmax=np.pi, interpolation='nearest', extent=[0,len(phi)*50,0,len(phi)], aspect='auto')
        fig.colorbar(p,label=r'$\varphi_1 - \varphi_i$')
        
        # phi,eps,alpha,beta,k1,k2,t_max = param
        # title = 
        
        plt.title(r"$\phi_1$ = " + f"{phi[0]},$\phi_2$ = " + f"{round(phi[-1],3)}, N = {self.N}, " + r"$\alpha$ = " + f"{round(alpha,3)}, " + r"$\beta$ = " + f"{round(beta,3)}" + dop_title)
        plt.xlabel("t")
        plt.ylabel("N")
        pdf.savefig()
        plt.savefig(way+'png\\' + f'graph_{count+1}.png')
        plt.savefig(way+'svg\\' + f'graph_{count+1}.svg')
        plt.close(fig)
        plt.clf()


    #тепловая карта---------------------------------------------------------------------------------------------------

START_PHI=1
N_JOB = 8
HOT = 1


col_razb = 10
MAX_GRAPH = 50
eps = 0.1
eps_map = 0#1e-1
Max_time = 100


if __name__ == "__main__":
    start = time.time()

    tmp = razb_config()
    ors = Original_sist(p = tmp, fi = 1)
    # ors.dinamic_klast(params=[[6.283185, 1.427449, 2, 1, 1.0471975511965976]])
    ors.sost_in_fi(key='st') #"st","un_st","rz","all"
    
    end = time.time()
    print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")
    
      
    # tmp = ['st','un_st','rz']
    
    # for i in tmp:
    #     ors.sost_in_fi(key=i) #"st","un_st","rz","all"
    
    

    # np.angel(fin - fi0)
    # параметр порядка

    # посмотреть 2х кластерное разбиение но со второй гармоникой