import numpy as np
from zamena import Equilibrium_states as Reduc
from original_sist import Original_sist as Orig
from matplotlib import pyplot as plt
from scipy.optimize import root
from tmp import heat_map as hm
import joblib 
import time
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit 
import os, shutil, sys
from scipy import interpolate
from scipy.integrate import solve_ivp
from supp_func import *
from tmp import heat_map
import random



class Tongue(Reduc,Orig):
    def __init__(self, p, par, h=1):
        Reduc.__init__(self, p[0:2])
        Orig.__init__(self, p, 1)
        self.param = par
        self.h = h
        self.start_sost = 0
        self.start_eig = 0
        self.t = np.linspace(0,Max_time,Max_time)
        self.M,self.alpha,self.beta,self.k1,self.k2 = par[1:]

        self.start_sost = [par[0],0]
        self.orig_way = f"{par}"

    def change_self(self,par):
        self.M,self.alpha,self.beta,self.k1,self.k2 = par[1:]
        self.start_sost = [par[0],0]

    def anti_par_zam(self, x):
        ress = []
        fi1 = self.fi1
        fi2 = fi1 - x[0]
        ress.append([fi1, fi2, self.M, self.alpha, self.beta, self.k1, self.k2])
        return ress

    def analyse_ord(self,arr):
        last_sost = arr[0]
        sign = 0
        count = 0
        last_s = arr[-1]
        min_s = arr[0]
        for x in arr:

            if min_s>x:
                min_s=x

            if last_sost - x > 0.1:
                if sign<0:
                    count+=1
                sign = 1
            else:
                if sign > 0.1:
                    count+=1
                sign = -1
            if count>=count_gran:
                return 'koleb'
            last_sost = x

        lam = self.eigenvalues([self.start_sost[0],self.M,self.alpha,self.beta,self.k1,self.k2])
        v = 'ust'
        for l in lam:
            if l.real > 0:
                v = 'ne_ust'
                break

        if min_s < 0.4 and last_s>0.95:
            return 'one'
        
        if last_s<0.95:
            return 'two_'+v
        
        # if last_s>0.95:
        #     return 'one'

        # if min_s>0.8:
        #     return 'one'

        return 'two_'+v
    
    def __find_sost_ravn__(self,start_point):
        x,v = start_point
        sol = root(self.syst_root,[x,v],method='lm')
        return sol.x
    
    def __bliz_sost__(self,s1,s2):
        return np.abs(s1[0]-s2[0])<ogr_sost
        

    def tmp(self):
        # start_point = [0.866757,0]
        # self.M, self.alpha,self.beta,self.k1,self.k2 = self.param[1:]
        # self.alpha+=0.1
        # tmp = self.__find_sost_ravn__(start_point)
        # print(tmp)
        par = self.anti_par_zam(self.param)
        par = np.reshape(par, (len(par[0])))

        start_point=np.zeros(4)
        start_point[0],start_point[1] = par[0:2] 
        start_point[0] = start_point[0]+eps
        start_point[1] = start_point[1]+eps
        start_point[2] = eps
        start_point[3] = eps

        print(self.__ord_par_tong__(start_point))

    def __ord_par_tong__(self,start_point,show=0,t=0):
        if type(t) == type(1) :
            t = self.t
        res = solve_ivp(self.syst_orig, [0,Max_time], start_point, t_eval=t)
        R1 = self.order_parameter(res.y)
        verdict = self.analyse_ord(R1)
        if (SHOW==1 and verdict=='koleb') or show==1 : 
            self.show_graph(R1,res.t)
        # print(start_point,self.alpha,self.beta,verdict)
        return verdict
     
    def point_analyse(self,sost,show=0,t=0):
        par = [sost[0],self.M,self.alpha,self.beta,self.k1,self.k2]
        lam = self.eigenvalues(par)
        start_point  = self.up_arr(1,par,self.N,self.N)
        for i in range(len(start_point)):
            start_point = np.append(start_point,0)
        lam_full = self.eigenvalues_full(start_point)
        lam_klast = []

        enum = 0
        for lf in lam_full:
            if enum<2:
                if np.abs(lf - lam[enum]) < 0.1:
                    enum+=1
                    continue
            
            if np.abs(lf)<1e-7:
                continue
            lam_klast.append(lf)

        res = ''
        # между кластерами
        f = True
        for l in lam:
            if l.real>EPS_REAL:
                f = False
                break
        if f:
            res+='st_'
        else:
            res+='unst_'
        # между кластерами
            
        # кластер
        f = True    
        for l in lam_klast:
            if l.real>0:
                f = False
                break
        if f:
            res+='st'
        else:
            res+='unst'
        # кластер

        if show == 1:
            print(lam)
            print(lam_klast)
            print(lam_full)

        return res
    
    def get_max_eig(self,sost):
        par = [sost[0],self.M,self.alpha,self.beta,self.k1,self.k2]
        start_point  = self.up_arr(1,par,self.N,self.N)
        for i in range(len(start_point)):
            start_point = np.append(start_point,0)
        lam_full = self.eigenvalues_full(start_point)
        max_ = lam_full[0].real
        for l in lam_full:
            if max_<l.real:
                max_ = l.real
        return max_

    def line_analyse(self,start_sost):
        res = []
        self.alpha,self.beta = start_sost[0:2]
        sost = self.start_sost
        while self.alpha>0:
            self.alpha -= h_alpha
            new_sost = self.__find_sost_ravn__(sost)
            if not self.__bliz_sost__(new_sost,sost):
                break
                res.append([self.alpha,self.beta,'False'])
            sost = new_sost
            # print([sost,self.alpha,self.beta])
            res.append([self.alpha,self.beta,self.point_analyse(sost),sost[0]])
        
        sost = self.start_sost
        self.alpha = self.param[2]
        while self.alpha<np.pi:
            self.alpha += h_alpha
            new_sost = self.__find_sost_ravn__(sost)
            if not self.__bliz_sost__(new_sost,sost):
                break
                res.append([self.alpha,self.beta,'False'])
            sost = new_sost
            # print([sost,self.alpha,self.beta])
            res.append([self.alpha,self.beta,self.point_analyse(sost),sost[0]])
        self.alpha = self.param[2]

        return res
    
    def __find_vert_sost__(self):
    
        vert_sost = []
        sost = self.start_sost
        while self.beta>0:
            self.beta -= h_beta
            new_sost = self.__find_sost_ravn__(sost)
            if not self.__bliz_sost__(new_sost,sost):
                break
                vert_sost.append([self.alpha,self.beta,'False'])
            sost = new_sost
            # print([sost,self.alpha,self.beta])
            vert_sost.append([self.alpha,self.beta,self.point_analyse(sost),sost[0]])
        
        sost = self.start_sost
        self.beta = self.param[3]
        while self.beta<np.pi:
            self.beta += h_beta
            new_sost = self.__find_sost_ravn__(sost)
            if not self.__bliz_sost__(new_sost,sost):
                break
                vert_sost.append([self.alpha,self.beta,'False'])
            sost = new_sost
            # print([sost,self.alpha,self.beta])
            vert_sost.append([self.alpha,self.beta,self.point_analyse(sost),sost[0]])
        self.beta = self.param[3]

        return vert_sost
    

    def all_analyse(self):
        vert_points = self.__find_vert_sost__()

        if ONCE_VERT==1:
            res = [vert_points]
        else:
            res = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.__work__)(s,vert_points) for s in vert_points)

        # for s in vert_points[0:10]:
        #     print(i)
        #     res.append(self.line_analyse(s))
        #     i+=1
        return res
    
    def __work__(self,s,vert_points):
        print(len(vert_points),": ",vert_points.index(s))
        return self.line_analyse(s)
        
    def max_eig_line(self,param):
        res = []
        self.alpha,self.beta = param[2:4]
        sost = [param[0],0]
        while self.alpha>0:
            self.alpha -= h_alpha
            new_sost = self.__find_sost_ravn__(sost)
            if not self.__bliz_sost__(new_sost,sost):
                break
            sost = new_sost
            # print([sost,self.alpha,self.beta])
            res.append([self.alpha,self.get_max_eig(sost)])
        
        sost = [param[0],0]
        self.alpha = param[2]
        while self.alpha<np.pi:
            self.alpha += h_alpha
            new_sost = self.__find_sost_ravn__(sost)
            if not self.__bliz_sost__(new_sost,sost):
                break
                res.append([self.alpha,self.beta,'False'])
            sost = new_sost
            # print([sost,self.alpha,self.beta])
            res.append([self.alpha,self.get_max_eig(sost)])

        return res

    
    def save_all_analyse(self,arr,save = False):
        if save:
            way= f"2_garmonic\\res\\n_{self.N}\\tmp\\{self.orig_way}\\"
            self.create_path(way+'pdf')
            self.create_path(way+'svg')
            self.create_path(way+'png')
            
            with open(way+'text.txt','w') as f:
                f.write(str(arr))
        
        tmp = []
        
        st_st = []
        unst_unst = []
        st_unst = []
        unst_st = []

        
        # arr = np.array(arr)
        for a in arr:
            # tmp = np.array(a)
            tmp = a
            for t_ in tmp:
                if t_[2] == 'st_st':
                    st_st.append([float(t_[0]),float(t_[1]),float(t_[3])])
                elif t_[2] == 'unst_unst':
                    unst_unst.append([float(t_[0]),float(t_[1]),float(t_[3])])
                elif t_[2] == 'st_unst':
                    st_unst.append([float(t_[0]),float(t_[1]),float(t_[3])])
                elif t_[2] == 'unst_st':
                    unst_st.append([float(t_[0]),float(t_[1]),float(t_[3])])
        st_st = np.array(st_st)
        st_st = st_st.T

        unst_unst = np.array(unst_unst)
        unst_unst = unst_unst.T

        st_unst = np.array(st_unst)
        st_unst = st_unst.T

        unst_st = np.array(unst_st)
        unst_st = unst_st.T
        
        if ALL == 1:
            s = 10
            if len(unst_st)!=0:
                plt.scatter(unst_st[0],unst_st[1],c='y',alpha=ALPHA,label = 'unst_st',s=s)
            
            if len(st_st)!=0:
                plt.scatter(st_st[0],st_st[1],c='b',label = 'st_st',s=s,alpha = ALPHA)
            
            if len(unst_unst)!=0:
                plt.scatter(unst_unst[0],unst_unst[1],c='g',alpha=ALPHA,label = 'unst_unst',s=s) 
            
            if len(st_unst)!=0:
                plt.scatter(st_unst[0],st_unst[1],c='r',alpha=ALPHA,label = 'st_unst',s = s)
            
            plt.title(f"N = {self.N}, M = {self.M}")
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\beta$")
            plt.legend()
            if ON_LIM:
                plt.xlim(-0.1,np.pi+0.2)
                plt.ylim(-0.1,np.pi+0.2)
            if save:
                plt.savefig(way + f'{self.orig_way}_N={self.N}.png')
                plt.savefig(way + f'{self.orig_way}_N={self.N}.svg')
            plt.show()   
        else:
            if len(st_unst)!=0:
                plt.scatter(st_unst[0],st_unst[1],c='r',alpha=ALPHA,label = 'st_unst',s = s)
                plt.xlim(-0.1,np.pi+0.2)
                plt.ylim(-0.1,np.pi+0.2)
                plt.legend()
                plt.show()
                st_unst = st_unst.T
                heat_map([st_unst[0][2],self.M,st_unst[0][0],st_unst[0][1],self.k1,self.k2])
            # if TWO == 1:
            if len(unst_unst)!=0:
                plt.scatter(unst_unst[0],unst_unst[1],c='g',alpha=ALPHA,label = 'unst_unst',s=s)
                plt.xlim(-0.1,np.pi+0.2)
                plt.ylim(-0.1,np.pi+0.2)
                plt.legend()
                plt.show()
                unst_unst = unst_unst.T
                heat_map([unst_unst[0][2],self.M,unst_unst[0][0],unst_unst[0][1],self.k1,self.k2])
            # if TWO == 1: 
            if len(unst_st)!=0:
                plt.scatter(unst_st[0],unst_st[1],c='y',alpha=ALPHA,label = 'unst_st',s=s)
                plt.xlim(-0.1,np.pi+0.2)
                plt.ylim(-0.1,np.pi+0.2)
                plt.legend()
                plt.show()
                unst_st = unst_st.T
                heat_map([unst_st[0][2],self.M,unst_st[0][0],unst_st[0][1],self.k1,self.k2])

            # if ONE == 1:
            if len(st_st)!=0:
                plt.scatter(st_st[0],st_st[1],c='b',label = 'st_st',s=s,alpha = ALPHA)
                plt.xlim(-0.1,np.pi+0.2)
                plt.ylim(-0.1,np.pi+0.2)
                plt.legend()
                plt.show()
                st_st = st_st.T
                heat_map([st_st[0][2],self.M,st_st[0][0],st_st[0][1],self.k1,self.k2])


    def show_graph(self, arr, time):
        plt.scatter(time, arr,s=5)
        plt.ylim(-0.01, 1.1)
        plt.xlabel("t")
        plt.ylabel('z')
        plt.title('order parametr')
        plt.show()

    def save_hot_map(self,arr,way):
        start_phi = START_PHI
        # dop_title = f', predict = '
        dop_title = ''
        joblib.Parallel(n_jobs = N_JOB3)(joblib.delayed(self.__work_hot_plot__)(x,arr,start_phi,way,dop_title) for x in arr)
        
        

N_JOB = 8
N_JOB3 = 3

Max_time = 100
count_gran = 10
eps = 1e-7



# =======LINE=======================================
def line():
    tmp = razb_config()
    par = PAR
    tong = Tongue(tmp,par)

    line_eig = tong.max_eig_line(par)
    line_eig = np.array(line_eig)
    line_eig = line_eig.T
    tong.show_graph(line_eig[1],line_eig[0])
# =======LINE=======================================

# =====ALL=========================================
def all_():

    tmp = razb_config()
    par = PAR#[2.892549, 2, 1.0472, 2.0944, 1, 1]
    tong = Tongue(tmp,par)

    reshenie = input("1: построение сначала, 2: построение по text\n")
    if reshenie == '1':    
        tmp1 = tong.all_analyse()
        good_arr = get_good_arr(tmp1)
        print(tong.save_all_analyse([good_arr]))
        reshenie = input("1: continue, es: save and exit, exit: exit\n")
    elif reshenie == '2':
        good_arr = razb_text(WAY)
        print(tong.save_all_analyse([good_arr]))
        reshenie = input("1: continue, es: save and exit, exit: exit\n")
    else:
        exit()
    
    while True:
        if reshenie == 'es':
            tong.save_all_analyse([good_arr],True)
            exit()
        if reshenie == 'exit':
            exit()
    # arr = razb_text(WAY)
        point = input("input point:\n").split()
        if len(point)!=2:
            break
        for i,p in enumerate(point):
            point[i]= float(p)
        ind = get_ind_text(good_arr,point)
        if ind==None:
            print("not in array")
            continue
        else:
            print(good_arr[ind])
        
        par[0] = good_arr[ind][3]
        par[2] = good_arr[ind][0]
        par[3] = good_arr[ind][1]

        tong.change_self(par)
        tmp2 = tong.all_analyse()
        good_arr_tmp2 = get_good_arr(tmp2)
        good_arr_tmp = add_unic_good_arr(good_arr,good_arr_tmp2)
        print(tong.save_all_analyse([good_arr_tmp]))
        
        reshenie = input("1: contionue, o: otkat, oe: otkat, save_graph and exit, e: save_graph and exit, exit: just exit\n")
        if reshenie == '1':
            good_arr = good_arr_tmp
            continue
        elif reshenie == 'o':
            tong.save_all_analyse([good_arr])
            continue
        elif reshenie == 'oe':
            tong.save_all_analyse([good_arr])
            break
        elif reshenie == 'e':
            tong.save_all_analyse([good_arr_tmp],True)
            break
        elif reshenie == 'exit':
            break
        

    
    
    

    # print(tong.all_analyse(par))
# =====ALL=========================================


# ====FIND KOLEB==========================================
def find_koleb():
    tmp = razb_config()
    par = PAR
    tong = Tongue(tmp,par)

    sost = SOST

    all_par = []
    with open(f"2_garmonic\\res\\n_{tmp[0]}\\"+sost+f"{tmp[0]}.txt") as file:
            for line in file:
                all_par.append(tong.razb_str(line.rstrip()))
    
    ind_koleb = []
    ind_one = []
    ind_two = []
    for i in range(len(all_par)):
        tong.change_self(all_par[i])
        verd = tong.point_analyse(all_par[i],0)
        if verd=='koleb':
            ind_koleb.append(i)
        if verd=='one':
            ind_one.append(i)
        if verd=='two':
            ind_two.append(i)
        print(i+1,": ",verd)
        
    print('indexs koleb: ',np.array(ind_koleb) + 1)
    print('indexs one klust: ',np.array(ind_one ) + 1)
    print('indexs two klust: ',np.array(ind_two) + 1)
# ====FIND KOLEB==========================================

# ==== TMP ==========================================
def tmp(params):
    tmp = razb_config()
    par = params#[2.892549, 2, 1.0472, 2.0944, 1, 1]
    # par[2] = 0.25#PAR[2]
    # par[3] = 1#PAR[3]
    tong = Tongue(tmp,par)

    print(tong.point_analyse(par,1))
    heat_map(par)
    
# ==== TPM ==========================================

def get_point_sost(points):
    good_arr = razb_text(WAY)
    point_map = []
    for point in points:
        ind = get_ind_text(good_arr,point)
        point_map.append(good_arr[ind])
    return point_map




# ==== ANALYSE SOST ==========================================

def analyse_sost():
    tmp = razb_config()
    
    way = f"2_garmonic\\res\\n_{tmp[0]}\\tmp\\"+str(PAR)+"\\analyse_sost"
    par = PAR#[2.892549, 2, 1.0472, 2.0944, 1, 1]
    par[2] = 2.5#PAR[2]
    par[3] = 0.5#PAR[3]
    tong = Tongue(tmp,par)

    tong.dinamic_klast([par])

# ==== ANALYSE SOST ==========================================

# stable
# [1.392219, 4, 1.0472, 3.14159, 1, 1]
# [0.678476, 6, 1.0472, 2.0944, 1, 1]
# [2.857196, 7, 2.0944, 0.0, 1, 1]
# stable

#rz 
#[0.00101, 5, 3.14159, 1.0472, 1, 1]
#rz

#[0.756906, 1, 1.0472, 1.0472, 1, 1]


# отчет
# [1.594466, 4, 1.0472, 2.0944, 1, 1]
# [0.756906, 1, 1.0472, 1.0472, 1, 1]
# отчет

PAR = [1.609267, 2, 0.0, 2.0944, 1, 1]#[1.047198, 1, 0.0, 3.14159, 1, 1]
SOST = 'stable_'
# [0.45000000000000023, 2.044400000000001, 2.119934357071167]
# [0.45000000000000023, 0.7944000000000007, 1.4431829695150693]
# [0.16, 2.1543999999999985, 1.6751802893528247]
ogr_sost = 0.05 # <- подобрать
h_alpha = 0.01
h_beta = 0.01

START_PHI = 1   
ONE = 1
TWO = 1
SHOW = 0
SAVE_HOT_MAP = 0
SAVE = 0

ALPHA = 1
EPS_REAL = 1e-8
ONCE_VERT = 0

ON_LIM = True
ALL = 1
tmp_conf = razb_config()
WAY = f"2_garmonic\\res\\n_{tmp_conf[0]}\\tmp\\{PAR}\\"

if __name__ == "__main__":
    
    print("start")
    start = time.time()


    all_()

    # points = [[0.358, 1.966],[0.1, 1.997],[0.381, 2.63],[1.7, 2.5],[1, 1.476],[1.15, 0.886]]
    # points = [[0.09, 2.233],[0.09, 1.947],[0.94, 1.705],[0.751, 1.885],[0.381, 1.935]]
    # points = [[0.733, 1.867]]
    # sost = get_point_sost(points)
    # print(sost)
    # for point in sost:
    #     tmp([point[3],PAR[1],point[0],point[1],PAR[4],PAR[5]])
        
    # analyse_sost()
    

    end = time.time()
    print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")
    

    
    
    # all_par = []
    # with open("2_garmonic\\res\\n_6\\non_stable_6.txt") as file:
    #         for line in file:
    #             all_par.append(tong.razb_str(line.rstrip()))
    
    # for i in range(len(all_par)):
    #     print(i+1,": ",tong.ord_par_tong(all_par[i]))
    
    # print(tong.ord_par_tong(par))