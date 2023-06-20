from numba import jit, cuda

import numpy as np
from zamena import Equilibrium_states as Reduc
from original_sist import Original_sist as Orig
from matplotlib import pyplot as plt
from scipy.optimize import root
import joblib 
import time
from scipy.integrate import solve_ivp
import os, shutil, sys


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
    
    def syst(self,t,param):
        m=self.m
        alpha=self.alpha
        N=self.N
        K=self.K
        M = self.M

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
        self.alpha = -np.pi
        self.m = m
        
        eig_old, sost_ravn = self.__iter_sr_eig__()
        
        return sost_ravn
    
    #основной блок
    def work(self,m):
        print(m)
        self.N = 13
        self.K, self.M = self.param[2:4]
        self.alpha = -np.pi
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

    def create_path(self, way):
        if not os.path.exists(way):
            os.makedirs(way)

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

    def razb_str(self,str):
        all = []
        tmp = ''
        tmp_arr = []
        flag = False

        for c in str:
            if c==',' or c==']':
                tmp_arr.append(float(tmp))
                tmp = ''
                continue
            tmp+=c
        
        return tmp_arr
    
    def find_border_tongue(self,m_space, arr_par, way):
        border_arr = []
        for par in arr_par:
            tmp_arr = []
            self.param = par
            self.alpha = -np.pi
            t = time.time()
        
            self.start_sost = self.__get_start_sost__(m_space[0])

            koord = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.work)(m) for m in m_space)
            # koord = np.array(koord)
            print("time work: ", time.time() - t)
            for line in koord:
                if len(line) == 0:
                    continue
                last_koord = line[0]
                for point in line:
                    if last_koord[2] != point[2]:
                        tmp_arr.append(point[0:2])
                    last_koord = point
            tmp_arr=self.sortMerge(tmp_arr)
            border_arr.append(tmp_arr)

        with open(way,'w',encoding="utf-8") as file:
            for border in border_arr:
                for el in border:
                    string = str(el[0]) + ',' + str(el[1]) +']'+'\n'
                    file.write(string)
                file.write('--' + '\n')

        print(time.time() - t)

    def sort(self,arr):
        res = []
        c = 0
        # for i in range(len(arr[0])-1):
        while c != len(arr[0])-2:
            if abs(arr[1][c] - arr[1][c+1]) < 0.1:
                res.append([arr[0][c],arr[1][c]])
            else:
                res.append([arr[0][c+1],arr[1][c+1]])
                c+=1
            c += 1

        res = np.array(res)
        res = res.T
        return res

    def plot_border_tongue(self,arr_par,way,m_space):
        border_arr_from_file = []
        # for par in arr_par:
        res = []
        with open(way,'r') as file:
            for line in file:
                if line.rstrip() == '--':
                    border_arr_from_file.append(res)
                    res = []
                    continue
                res.append(np.array(self.razb_str(line.rstrip())))
        # print(res)
        # border_arr_from_file.append(res)
        
        color_arr = ['b','r','g','y']
        marker_arr = ['','--','-.',]
        for i in range(len(border_arr_from_file)):
            x = border_arr_from_file[i]
            x = np.array(x)
            x = x.T
            first_cluster = []
            second_cluster = []
            for j in range(len(x[0])):
                if x[0][j] <= 0:
                    first_cluster.append([x[0][j],x[1][j]])
                else:
                    second_cluster.append([x[0][j],x[1][j]])
            first_cluster = np.array(first_cluster)
            first_cluster = first_cluster.T
            second_cluster = np.array(second_cluster)
            second_cluster = second_cluster.T

            # new_second_cluster = self.sort(second_cluster)

            # plt.scatter(first_cluster[0],first_cluster[1],c='b',alpha=0.3)
            # plt.scatter(new_first_cluster[0],new_first_cluster[1],c='r',alpha=0.3)

            t1 = np.polyfit(first_cluster[0],first_cluster[1],7)
            f1 = np.poly1d(t1)
            t2 = np.polyfit(second_cluster[0],second_cluster[1],7)
            f2 = np.poly1d(t2)
            new_x = - first_cluster[0]

            
            # new_space = []
            # for j in range(len(new_x)):
            #     tmp_arr = np.linspace(first_cluster[0][i],new_x[i],round(100*m_space[j]),0)
            #     for k in range(len(tmp_arr)):
            #         new_space.append([tmp_arr[k],m_space[len(m_space)-j-1]])
            # res = self.non_ust_obl(arr_par[i],new_space)
            # plt.scatter(res[0],res[1],c=self.choose_colore(res[2]))

            plt.plot(new_x,f1(first_cluster[0]),marker_arr[i],c=color_arr[i],label=arr_par[i])
            plt.plot(first_cluster[0],f1(first_cluster[0]),marker_arr[i],c=color_arr[i])
            plt.xlabel(r'$\alpha$')
            plt.ylabel('m')
            plt.grid()
            # plt.plot(new_second_cluster[0],f2(new_second_cluster[0]),c='y',alpha=0.5)
            # plt.scatter(second_cluster[0],second_cluster[1],c='b',alpha=0.3)
            # plt.scatter(new_second_cluster[0],new_second_cluster[1],c='r',alpha=0.3)
            
            
            # plt.plot(new_x, f2(new_second_cluster[0]),c='r',alpha=0.5)
            # args1, covar = curve_fit(self.mapping4, new_first_cluster[1], new_first_cluster[0]) 
            # a1, b1 = args1[0], args1[1]
            # args2, covar = curve_fit(self.mapping3, second_cluster[0], second_cluster[1]) 
            # a2, b2, c2, d2 = args2[0], args2[1], args2[2], args2[3]

            # y_fit1 = a1 + b1 * np.log(new_first_cluster[0])
            # y_fit2 = a2 * np.exp(b2*second_cluster[0]**2 + c2*second_cluster[0] + d2) 
            # plt.plot(new_first_cluster[0],y_fit1,alpha=0.5, c='y')
            # plt.scatter(first_cluster[0],first_cluster[1],c='b',alpha=0.5)
            
            # args1, covar = curve_fit(self.mapping2, first_cluster[0], first_cluster[1]) 
            # a1, b1, c1, d1, e1 = args1[0], args1[1], args1[2], args1[3], args1[4]
            # args2, covar = curve_fit(self.mapping2, second_cluster[0], second_cluster[1]) 
            # a2, b2, c2, d2, e2 = args2[0], args2[1], args2[2], args2[3], args2[4]

            # y_fit1 = a1 * first_cluster[0]**4 + b1 * first_cluster[0]**3 + c1 *first_cluster[0]**2 + d1*first_cluster[0] + e1
            # y_fit2 = a2 * second_cluster[0]**4 + b2 * second_cluster[0]**3 + c2 *second_cluster[0]**2 + d2*second_cluster[0] + e2
            
            # plt.plot(first_cluster[0],y_fit1,c=color_arr[i],alpha=0.5)
            # plt.plot(second_cluster[0],y_fit2,c=color_arr[i],alpha=0.5,label=arr_par[i])
            # plt.xlabel(r'$\alpha$')
            # plt.ylabel('m')

            # plt.scatter(first_cluster[0],first_cluster[1],c='y',alpha=0.3)
            # plt.scatter(second_cluster[0],second_cluster[1],c='y',alpha=0.3)
            # убираем высеры -------------------------------------------------------------------------------------------------

            # new_first_cluster = []
            # for k in range(len(first_cluster[1])):
            #     if first_cluster[1][k]-y_fit1[k] < 0.5:
            #             new_first_cluster.append([first_cluster[0][k],first_cluster[1][k]])
            # new_second_cluster = []
            # for k in range(len(second_cluster[1])-1):
            #     if second_cluster[1][k] < second_cluster[1][k+1] + 0.01 and second_cluster[0][k] < second_cluster[0][k+1] + 0.01:  #or abs(second_cluster[1][k]-y_fit2[k])<0.3
            #         new_second_cluster.append([second_cluster[0][k],second_cluster[1][k]])
            # new_first_cluster = np.array(new_first_cluster)
            # new_first_cluster = new_first_cluster.T
            # new_second_cluster = np.array(new_second_cluster)
            # new_second_cluster = new_second_cluster.T
            # args1, covar = curve_fit(self.mapping3, new_first_cluster[0], new_first_cluster[1]) 
            # a1, b1, c1= args1[0], args1[1], args1[2]
            # args2, covar = curve_fit(self.mapping3, new_second_cluster[0], new_second_cluster[1]) 
            # a2, b2, c2 = args2[0], args2[1], args2[2]
            # y_fit1_new = a1 * np.exp(b1*new_first_cluster[0] + c1) 
            # y_fit2_new = a2 * np.exp(b2*new_second_cluster[0] + c2) 
            # # plt.plot(new_first_cluster[0],y_fit1_new, c='r',alpha=0.5)
            # plt.plot(second_cluster[0],y_fit2,c='b',alpha=0.5)
            # plt.plot(new_second_cluster[0],y_fit2_new, c='r',alpha=0.5)
            # plt.scatter(second_cluster[0],second_cluster[1],c='b',alpha=0.5)
            # plt.scatter(new_second_cluster[0],new_second_cluster[1],c='r',alpha=0.5)
            
            # ---------------------------------------------------------------------------------------



        plt.legend()
        plt.show()
        
            # joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.plot_eig_lvl)(m,par,h,proc) for m in [m_space[0],m_space[len(m_space)//2],m_space[len(m_space)-1]])
        
        
        # for m in [m_space[0],m_space[len(m_space)//2],m_space[len(m_space)-1]]:
        #     self.plot_eig_lvl(m, par,1e-5)

    def plot_three_lvl_eig(self,m_space , param ,proc, h):
        for par in param:
            joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.plot_eig_lvl)(m,par,h,proc) for m in [m_space[0],m_space[len(m_space)//2],m_space[len(m_space)-1]])
        # for m in [m_space[0],m_space[len(m_space)//2],m_space[len(m_space)-1]]:
        #     self.plot_eig_lvl(m, par,1e-5)
    
    
    def mapping3(self, values_x, a, b, c): 
        return a * np.exp(b*values_x + c)
    
    def mapping4(self, values_x, a, b): 
        return a + b * np.log(values_x)
    
    def sravn(self, ar1,ar2):
        arr = []
        i1 = 0
        i2 = 0
        while i1 != len(ar1) and i2 != len(ar2):
            
            if ar1[i1][0] <= ar2[i2][0]:
                arr.append(ar1[i1])
                i1+=1
            else:

                arr.append(ar2[i2])
                i2+=1
    
        while i1 != len(ar1):
            arr.append(ar1[i1])
            i1+=1
        while i2 != len(ar2):
            arr.append(ar2[i2])
            i2+=1
            
        return arr       

    def sortMerge(self, arr):
        if len(arr) <= 1:
            return arr
        l = self.sortMerge(arr[0:len(arr)//2])
        r = self.sortMerge(arr[len(arr)//2:len(arr)])
        return self.sravn( l , r )

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
            
            space = round(len(koord)/round(len(koord)*(proc)))
            # space = round(163/80)
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
    @jit(nopython=False, parallel=True)
    def func_to_paral(self,m,alpha,par):
        self.m = m
        self.alpha = alpha
        time = np.linspace(0,100,100)
        res = solve_ivp(self.syst,[0, 100], [par[0],par[1],0.01,0.01], t_eval=time)
        new_x = []#res.y[0][len(res.y[0])//2]
        new_y = []
        for i in range(len(res.y[0])//2,len(res.y[0])):
            new_x.append(res.y[0][i])
            new_y.append(res.y[1][i])
        max_x = max(new_x)
        max_y = max(new_y)
        min_x = min(new_x)
        min_y = min(new_y)
        last_x = np.angle(np.exp(1j*new_x[-1]))
        last_y = np.angle(np.exp(1j*new_y[-1]))
        flag = 0
        if abs(max_x-min_x)<1e-3: #x - константа -> либо синфаза, либо двухкластер.
            if abs(last_x)<1e-3: #проверка на синфазу
                if abs(max_y-min_y)<1e-3: # y - константа
                    if abs(last_y)<1e-3:
                        flag = 1 # это синфаза
                    else:
                        flag = 2 # двухкластерное
                else:
                    flag = 3 #двухкластерное с колебаниями
            else: 
                if abs(max_y-min_y)<1e-3:
                    if abs(last_y)<1e-3:
                        flag = 2 # двухкластерное
                    else:
                        flag = 4 #трехкластерное
                else:
                    flag = 5 #трехкластерное с колебаниями
        else:
            if abs(max_y-min_y)<1e-3:
                if abs(last_y)<1e-3:
                    flag = 3 #двухкластерное с колебаниями
                else:
                    flag = 5 #трехкластерное с колебаниями
            else:
                flag = 6 #хаос
        return [m,alpha,flag]

    @jit(nopython=False, parallel=True)
    def non_ust_obl(self,par,new_space):
        new_space = np.array(new_space)
        new_space = new_space.T
        m_space = new_space[1]
        al_space = new_space[0]
        res_arr=[]
        # res_arr = joblib.Parallel(n_jobs = 6)(joblib.delayed(self.func_to_paral)(m,alpha,par) for m in m_space for alpha in al_space)
        for alpha in al_space:
            for m in m_space:
                res_arr.append(self.func_to_paral(m,alpha,par))
        res_arr = np.array(res_arr)
        res_arr = res_arr.T
        return res_arr
    
    def choose_colore(self,flag):
        c = 'w'
        if flag==1:
            c = 'g'
        elif flag == 2:
            c = 'c'
        elif flag == 2:
            c = 'm'
        elif flag == 3:
            c = 'y'
        elif flag == 4:
            c = 'purple'
        elif flag == 5:
            c = 'orange' 
        elif flag == 6:
            c = 'k'
        return c

        
        
eps = 1e-7
ogr_sost = 1e-3
N_JOB = 6
# h_eps = 0.01

if __name__ == "__main__":
    tmp = [13, 1, 1]
    par = [4.459709, 2.636232, 2, 1, 2.0944]#[4.75086, 4.75086, 1, 3, 2.0944]     [4.459709, 2.636232, 2, 1, 2.0944]
    arr_par = [[1.654226, 3.308453, 6, 1, 2.0944]] #[4.459709, 2.636232, 2, 1, 2.0944], [1.823477, 4.459709, 1, 2, 4.18879]
    h = 1e-4
    tong = Tongue(tmp,par,h)
    way = f"new_life\\res\\n_{tong.N}\\border_tongue_13.txt"
    # way = "border_tongue_2.txt"
    # way_tmp = f"new_life\\res\\n_{tong.N}\\border_tongue"
    # print(tong.tmp(1.8421052631578947))
    # tong.tmp(1, 2.0944)
    m_space = np.linspace(0.1,10,100)
    # tong.find_tongue(m_space)
    tong.find_border_tongue(m_space,arr_par,way)#,1e-3,0.5)
    # tong.plot_border_tongue(arr_par,way,m_space)

    # tong.plot_three_lvl_eig(m_space,arr_par,1,1e-4)

    # tong.plot_eig_lvl(m_space[-1],[1.823477, 3.646953, 2, 1, 2.0944], 1e-6, 0.1) #самое правое значение от 0 до 1
    
    # tong.plot_eig(1,2.0944*2)
   