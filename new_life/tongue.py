import numpy as np
from zamena import Equilibrium_states as Reduc
from original_sist import Original_sist as Orig
from matplotlib import pyplot as plt
from scipy.optimize import root
from tmp import heat_map as hm
import joblib 
import time



class Tongue(Reduc,Orig):
    def __init__(self, p, par, h=1):
        Reduc.__init__(self, p[0:2])
        Orig.__init__(self, p, 1)
        self.param = par
        self.h = h
        self.start_sost = 0
        self.start_eig = 0

    def change_params(self,par):
        self.param = par
    #определение устойчивости состояния равновесия (больше для редуцированной системы)
    def sustainability_analysis(self,arr):
        pass

    def __syst_reg__(self,param):
        N = self.N
        M = self.M
        K = self.K
        alpha = self.alpha
        m = self.m
        
        x,y,v,w = param #[x,y,w,v] - точки
        f = np.zeros(4)
        # x with 1 dots
        f[1] = v
        # y with 1 dots
        f[2] = w
        # x with 2 dot
        f[2] = 1/m*(1/N * ((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha)) - v)
        # y with 2 dot
        f[3] = 1/m*(1/N * ((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-M*np.sin(y-x-alpha)) - w)
        
        return f
    #нахождение собственных значений
    def __eigenvalues__(self,sost):
        start_phi = 0
        arr = [sost[0], sost[1], self.K, self.M, self.alpha]
        start_phi = super().up_arr(start_phi,arr ,self.N,self.N)
        start_phi = np.append(start_phi,np.zeros(len(start_phi)))

        eig_redu = Reduc.eigenvalues(self,arr)
        eig_orig = Orig.eigenvalues_full(self,start_phi)

        return eig_orig,eig_redu
    #рисование собственных чисел
    def __paint_lams__(self, eig):
        
        eig_orig, eig_redu = eig

        old_lam = np.array(eig_orig)
        new_lam = np.array(eig_redu)
        real_deal_old = old_lam.real
        not_real_deal_old = old_lam.imag
        real_deal_new = new_lam.real
        not_real_deal_new = new_lam.imag

        plt.scatter(real_deal_new, not_real_deal_new, c='b', marker='o')
        plt.scatter(real_deal_old, not_real_deal_old, c='r', marker='x')
        plt.grid()
        plt.show()
    
    def plot_eig(self,m, alpha):
        self.N = 5
        self.K, self.M = self.param[2:4]
        self.alpha = self.param[4]
        self.m = m
        self.alpha = alpha
        
        eig, sost = self.__iter_sr_eig__()
        
        self.__paint_lams__(eig)
        
    
    #нахождение состояния равновесия
    def __find_sost_ravn__(self):
        
        start_point = np.array(self.param[0:2])
        start_point = np.append(start_point, np.zeros(len(start_point))) 
        # start_point = [2.636232, 4.459709, 0., 0.] #[2.636232, 4.459709, 0., 0.]
        x,y,v,w = start_point

        sol = root(self.__syst_reg__,[x,y,v,w],method='lm')

        return sol
    #итерация по поиску ср и его собст чисел
    def __iter_sr_eig__(self):
        sost = self.__find_sost_ravn__()
        eig = self.__eigenvalues__(sost.x)
        # self.__paint_lams__(eig)
        return [eig,sost]
    
    #поиск состояния в некоторой окрестности
    def __sr_in_e_okr__(self, old_eig,old_sost):
        num_h = 10
        
        eig,tmp_sost = self.__iter_sr_eig__()
        # f = self.__check_sost_ravn__(tmp_sost, old_sost)
        
        # if f:
        #     for i in range(num_h):
        #         self.param[0] = old_sost.x[0] + (1+i)*h_eps
        #         self.param[1] = old_sost.x[1] + (1+i)*h_eps
        #         eig,tmp_sost = self.__iter_sr_eig__()
        #         f = self.__check_sost_ravn__(tmp_sost, old_sost)        
        #         if not f:
        #             return [eig,tmp_sost]
        #     for i in range(num_h):
        #         self.param[0] = old_sost.x[0] - (1+i)*h_eps
        #         self.param[1] = old_sost.x[1] - (1+i)*h_eps
        #         eig,tmp_sost = self.__iter_sr_eig__()
        #         f = self.__check_sost_ravn__(tmp_sost, old_sost)        
        #         if not f:
        #             return [eig,tmp_sost]
        # else:
        #     return [eig,tmp_sost]

        # return [old_eig,old_sost]
        return [eig,tmp_sost]
        
                    
        
    
    #изменение шага параметра
    def __change_h__(self,eig_new,eig_old, ind):
        
        if np.abs(eig_new[ind].real) < eps:
            return 0
        
        if eig_old[ind].real >= 0:
            if eig_new[ind].real < 0:
                self.h = -self.h/2
        else:
            if eig_new[ind].real > 0:
                self.h = -self.h/2
            
        
        return 1
        # for e in range(len(eig_new)):
        #     if eig_old

    #получение индекса приближенного с.ч. к 0
    def __get_index__(self,eig):
        ind = 0
        tmp = 0
        res = []
        for e in eig:
            if np.abs(e.real) < np.abs(eig[tmp].real):
                tmp = ind
                
            ind += 1
        # ind = 0
        # for e in eig:
        #     if np.abs(e.real) == np.abs(eig[tmp].real):
        #        res.append(ind)
        #     ind+=1 
        
        return tmp
        # return res
    
    #проверка совпадения с.ч. редуц и ориг систем
    def __ne_sopost_eig__(self,eig):
        f = False
        
        for i in range(len(eig[1])):
            for j in range(len(eig[0])):
                if np.abs(eig[1][i]-eig[0][j]) < 1e-2:
                    f = True
            if f:
                f= False
            else:
                return True
            
        return False

    #проверка перехода в другое состояние
    def __check_sost_ravn__(self,old,new):
        f = False
        for i in range(len(old.x)):
            tmp = np.abs(old.x[i]-new.x[i])
            if(tmp)>ogr_sost:
                f= True
                break
        return f
    
    #проверка устойчивости состояния (True - неустойчивое)
    def __check_eig__(self,eig):
        f = False
        for i in eig:
            if i.real>0:
                f = True
                break
        return f
    
    def __max_eig__(self,eig):
        max_eig = -1000
        
        for i in eig:
            if i.real>max_eig:
                max_eig = i.real
                comp = isinstance(i, complex)
                
        return max_eig, comp

    #функция для проверок
    def tmp(self,m, alpha):
        self.N = 5
        self.K, self.M = self.param[2:4]
        self.alpha = self.param[4]
        sost_ravn = []
        self.m = m
        self.alpha = alpha
        
        tmp = self.__get_start_sost__(m)
        
        
        self.work(m,tmp)
        plt.show()
        # # eig_old, sost_ravn = self.__iter_sr_eig__()
        # # index = self.__get_index__(eig_old[1])
        
        # # print(sost_ravn.x)
        
        # # self.__paint_lams__(eig_old)
        # return self.__check_eig__(eig_old[1])
    
    
    #стартовое состояние
    def __get_start_sost__(self,m):
        self.N = 5
        self.K, self.M = self.param[2:4]
        self.alpha = self.param[4]
        self.m = m
        
        eig_old, sost_ravn = self.__iter_sr_eig__()
        
        return sost_ravn
    
    #основной блок
    def work(self,m):
        print(m)
        self.N = 5
        self.K, self.M = self.param[2:4]
        self.alpha = self.param[4]
        sost_ravn = []
        self.m = m
        
        koord = []
        sost_last = self.start_sost
        eig_start = self.start_eig
        
        start_alpha = self.alpha
        
        sost_last.x = self.param[0:2]

        #состояние на 0 шаге --------------------------
        eig_old, sost_ravn = self.__sr_in_e_okr__(eig_start, sost_last)
        if self.__check_sost_ravn__(sost_last,sost_ravn) or sost_ravn.success==False or self.__ne_sopost_eig__(eig_old):
            pass
        else:
            # index = self.__get_index__(eig_old[1])
            sost_ravn_old = sost_ravn
            koord.append([self.alpha,self.m,self.__check_eig__(eig_old[1])])
            #состояние на 0 шаге --------------------------
            
            
            f = 1
            # tmp = 0
            while f:
                self.alpha += self.h
                eig_new,sost_ravn = self.__sr_in_e_okr__(eig_old, sost_ravn)
                if  self.__check_sost_ravn__(sost_ravn_old,sost_ravn) or sost_ravn.success==False or self.__ne_sopost_eig__(eig_new) or np.abs(self.alpha) > np.pi:
                    if np.abs(self.alpha) < np.pi:
                        continue
                    self.alpha = start_alpha
                    break
                    # self.h = -self.h
                    # tmp+=1
                    # if tmp ==2:
                    # continue
                
                eig_old = eig_new
                koord.append([self.alpha,self.m,self.__check_eig__(eig_new[1])])
                # if (self.alpha < 2.834399999999984 + 0.1 and self.alpha > 2.834399999999984 - 0.1) and (self.m <1.8421052631578947+0.01 and self.m > 1.8421052631578947-0.01):
                #     self.__paint_lams__(eig_new)
                sost_ravn_old = sost_ravn
                self.param[0:2] = sost_ravn.x[0:2]
                

            koord = np.array(koord)
            
            eig_old, sost_ravn = self.__iter_sr_eig__()
            self.start_sost = sost_ravn
            self.start_eig = eig_old
        
        return koord        

    def find_border_tongue(self,m_space, arr_par,h,proc):
        border_arr = []
        for par in arr_par:
        #     tmp_arr = []
        #     self.param = par
        #     self.alpha = -np.pi
        #     t = time.time()
        
        #     self.start_sost = self.__get_start_sost__(m_space[0])
        #     # for m in m_space:
        #     #     start_sost = self.work(m,start_sost)

        #     koord = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.work)(m) for m in m_space)
        #     # print(self.sost())
        #     koord = np.array(koord)
        #     print("time work: ", time.time() - t)
        #     for line in koord.T:
        #         if len(line) == 0:
        #             continue
        #         last_koord = line[0]
        #         for point in line:
        #             if last_koord[2] != point[2]:
        #                 tmp_arr.append(point[0:2])
        #             last_koord = point
        #     border_arr.append(tmp_arr)

        # color_arr = ['b','r','g','y']
        # for i in range(len(border_arr)):
        #     x = border_arr[i]
        #     x = np.array(x)
        #     x = x.T
        #     plt.scatter(x[0],x[1],c=color_arr[i], alpha = 0.5)
            
        # print(time.time() - t)
        # plt.show()
        
            joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.plot_eig_lvl)(m,par,h,proc) for m in [m_space[0],m_space[len(m_space)//2],m_space[len(m_space)-1]])
        
        
        # for m in [m_space[0],m_space[len(m_space)//2],m_space[len(m_space)-1]]:
        #     self.plot_eig_lvl(m, par,1e-5)

    def find_tongue(self,m_space):
        t = time.time()
        
        self.start_sost = self.__get_start_sost__(m_space[0])
        # for m in m_space:
        #     start_sost = self.work(m,start_sost)

        koord = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.work)(m) for m in m_space)
        # print(self.sost())
        # koord = np.array(koord)
        print("time work: ", time.time() - t)

        t = time.time()
        
        ust = []
        ne_ust = []
        
        for j in koord:
            for i in j:
                if i[2] == 1:
                    ust.append([i[0:2]])
                else:
                    ne_ust.append([i[0:2]])
        ust = np.array(ust)
        ne_ust = np.array(ne_ust)
        
        ust = ust.T
        ne_ust = ne_ust.T
        if len(ust>0):
            plt.scatter(ust[0],ust[1],c='r')
        if len(ne_ust>0):
            plt.scatter(ne_ust[0],ne_ust[1],c='b')
        
        # print(time.time() - t)
        plt.show()


    def __find_max_eig__(self,m, sost_last,h,proc):
        self.N = 5
        self.K, self.M = self.param[2:4]
        sost_ravn = []
        
        koord = []
        print("nachal")
        
        start_alpha = self.alpha
        
        #состояние на 0 шаге --------------------------
        eig_old, sost_ravn = self.__iter_sr_eig__()
        if self.__check_sost_ravn__(sost_last,sost_ravn) or sost_ravn.success==False or self.__ne_sopost_eig__(eig_old):
            pass
        else:
            # index = self.__get_index__(eig_old[1])
            sost_ravn_old = sost_ravn
            
            
            eig_compl = self.__max_eig__(eig_old[1])            
            koord.append([self.alpha,eig_compl[0],self.__check_eig__(eig_old[1]),eig_compl[1]])
            #состояние на 0 шаге --------------------------
            
            
            f = 1
            # tmp = 0
            while f:
                self.alpha += h
                eig_new,sost_ravn = self.__sr_in_e_okr__(eig_old, sost_ravn)
                if  self.__check_sost_ravn__(sost_ravn_old,sost_ravn) or sost_ravn.success==False or self.__ne_sopost_eig__(eig_new):
                    if np.abs(self.alpha) < np.pi:
                        continue
                    self.alpha = start_alpha
                    break
                    # self.h = -self.h
                    # tmp+=1
                    # if tmp ==2:
                    # continue
                
                eig_old = eig_new
                
                eig_compl = self.__max_eig__(eig_old[1])
                koord.append([self.alpha,eig_compl[0],self.__check_eig__(eig_old[1]),eig_compl[1]])
                # if (self.alpha < 2.834399999999984 + 0.1 and self.alpha > 2.834399999999984 - 0.1) and (self.m <1.8421052631578947+0.01 and self.m > 1.8421052631578947-0.01):
                #     self.__paint_lams__(eig_new)
                sost_ravn_old = sost_ravn
                self.param[0:2] = sost_ravn.x[0:2]
                

            koord = np.array(koord)
            print("risyu")
            
            ust_real=[]
            ne_ust_real=[]
            ust_complex=[]
            ne_ust_complex=[]
            
            space = len(koord)//proc
            tmp_i = 0
            for i in koord:
                tmp_i+=1
                if tmp_i == space:
                    tmp_i = 0
                    
                    if i[2] == 1:
                        if i[3]==1:
                            ust_complex.append([i[0:2]])
                        else:
                            ust_real.append([i[0:2]])
                    else:
                        if i[3]==1:
                            ne_ust_complex.append([i[0:2]])
                        else:
                            ne_ust_real.append([i[0:2]])
                else:
                    continue
                
                        
                
            ust_real        = np.array(ust_real)
            ne_ust_real     = np.array(ne_ust_real)
            ust_complex     = np.array(ust_complex)
            ne_ust_complex  = np.array(ne_ust_complex)
            
            fig, ax = plt.subplots()
            
            ust_real        = ust_real.T
            ne_ust_real     =ne_ust_real.T
            ust_complex     =ust_complex.T
            ne_ust_complex  =ne_ust_complex.T
            
            if len(ust_real>0):
                ax.scatter(ust_real[0],ust_real[1],c='r',marker='x')
            if len(ne_ust_real>0):
                ax.scatter(ne_ust_real[0],ne_ust_real[1],c='b',marker='o')
            if len(ust_complex>0):
                ax.scatter(ust_complex[0],ust_complex[1],c='r',marker='x')
            if len(ne_ust_complex>0):
                ax.scatter(ne_ust_complex[0],ne_ust_complex[1],c='b',marker='o')

            plt.show()

    def plot_eig_lvl(self,m,param,h,proc):
        self.N = 5
        self.param = param
        self.K, self.M = self.param[2:4]
        self.m = m
        self.param[4] = -np.pi
        
        start_sost = self.__get_start_sost__(m)
        self.__find_max_eig__(m,start_sost,h,proc)
        
        plt.show()
        

        
        
eps = 1e-7
ogr_sost = 0.001
N_JOB = 8
# h_eps = 0.01

if __name__ == "__main__":
    tmp = [5, 1, 1]
    par = [4.459709, 2.636232, 2, 1, 1.0472] #[4.75086, 4.75086, 1, 3, 2.0944]	 [4.459709, 2.636232, 2, 1, 2.0944]
    arr_par = [[1.823477, 3.646953, 2, 1, 2.0944],[1.823477, 3.646953, 2, 1, 3.14159],[1.823477, 4.459709, 1, 2, 3.14159],[2.636232, 4.459709, 2, 2, 3.14159]] #[4.459709, 2.636232, 2, 1, 2.0944],
    h = 1e-2
    tong = Tongue(tmp,par,h)
    # print(tong.tmp(1.8421052631578947))
    # tong.tmp(1, 2.0944)
    m_space = np.linspace(0.1,50,500)
    # tong.find_tongue(m_space)
    tong.find_border_tongue(m_space,arr_par)
    # tong.plot_eig_lvl(1,2.0944)
    
    # tong.plot_eig(1,2.0944*2)
   