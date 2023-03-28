import numpy as np
from zamena import Equilibrium_states as Reduc
from original_sist import Original_sist as Orig
from matplotlib import pyplot as plt
from scipy.optimize import root


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


    #нахождение собственных значений
    def __eigenvalues__(self,sost):
        start_phi = 0
        arr = [sost[0], sost[1], self.K, self.M, self.alpha]
        start_phi = super().up_arr(start_phi,arr ,self.N,self.N)
        start_phi = np.append(start_phi,np.zeros(len(start_phi)))

        eig_redu = Reduc.eigenvalues(self,arr)
        eig_orig = Orig.eigenvalues_full(self,start_phi)

        return eig_orig,eig_redu
    
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
    
    def __find_sost_ravn__(self):
        
        start_point = np.array(self.param[0:2])
        start_point = np.append(start_point, np.zeros(len(start_point))) 
        # start_point = [2.636232, 4.459709, 0., 0.] #[2.636232, 4.459709, 0., 0.]
        x,y,v,w = start_point

        sol = root(super().syst,[x,y,v,w],args=(0),method='lm')

        return sol
        
    def __iter_work__(self):

        

        sost = self.__find_sost_ravn__()
        eig = self.__eigenvalues__(sost.x)
        self.__paint_lams__(eig)


    def work(self):
        self.N = 5
        self.K, self.M = self.param[2:4]
        self.alpha = self.param[4]
        
        self.__iter_work__()
        
        for i in range(10):
            self.alpha+= self.h
            self.__iter_work__()


        
        



        


if __name__ == "__main__":
    tmp = [5 ,1, 1]
    par = [2.636232, 4.459709, 2, 2, 1.0472] #[2.636232, 4.459709, 2, 2, 1.0472] 32
    h = 0.1
    tong = Tongue(tmp,par,h)
    tong.work()
