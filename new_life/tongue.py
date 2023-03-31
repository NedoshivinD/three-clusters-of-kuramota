import numpy as np
from zamena import Equilibrium_states as Reduc
from original_sist import Original_sist as Orig
from matplotlib import pyplot as plt
from scipy.optimize import root
from tmp import heat_map as hm

eps = 1e-7

class Tongue(Reduc,Orig):
    def __init__(self, p, par, h=1):
        Reduc.__init__(self, p[0:2])
        Orig.__init__(self, p, 1)
        self.param = par
        self.h = h
    
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
        # x with 1 dot
        f[0] = 1/m*(1/N * ((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha)) - v)
        # y with 1 dot
        f[1] = 1/m*(1/N * ((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-M*np.sin(y-x-alpha)) - w)
        # x with 2 dots
        f[2] = v
        # y with 2 dots
        f[3] = w
        
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
        

    def work(self):
        self.N = 5
        self.K, self.M = self.param[2:4]
        self.alpha = self.param[4]
        sost_ravn = []
        
        eig_old, sost_ravn = self.__iter_sr_eig__()
        index = self.__get_index__(eig_old[1])

        self.__paint_lams__(eig_old)
            
        
        f = 1
        while f:
            self.alpha+= self.h
            eig_new,sost_ravn = self.__iter_sr_eig__()
            f = self.__change_h__(eig_new[1],eig_old[1],index)
            # self.__paint_lams__(eig_old)
            eig_old = eig_new
            
        # print(it)
        self.__paint_lams__(eig_new)
        arr = [sost_ravn.x[0],sost_ravn.x[0],self.K,self.M,self.alpha]
        print(arr)
        print('eig_red: ', eig_new[1])
        print('eig_orig: ', eig_new[0])
        print('iter: ', arr)
        hm(arr)
        


        
        

if __name__ == "__main__":
    tmp = [5 ,1, 1]
    par = [2.636232, 4.459709, 2, 2, 1.0472] #[2.636232, 4.459709, 2, 2, 1.0472] 32 [2.636232, 4.459709, 2, 2, 1.0472]	
    h = -0.5
    tong = Tongue(tmp,par,h)
    tong.work()
