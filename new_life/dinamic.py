import numpy as np
from scipy.integrate import solve_ivp
import joblib 
import matplotlib.pyplot as plt
import os, shutil, sys

Max_time = 100
max_el = 50
N_JOB = 4
eps = 1.5



class Dinamic(object):
    
    #инициализация системы
    def __init__(self,p = [3,1, 1]):
        self.N, self.m, self.omega = p
        self.M = 2
        self.K = 2
        self.alpha = 0
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,Max_time,Max_time)

    def syst(self,t, param):
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
     
    def calculation(self, x, par, way):
        M = self.M
        K = self.K
        alpha = par
        way = way + "\\"
        start_point=np.zeros(4)
        start_point[0] = x[0]
        start_point[1] = x[1]
        start_point[2] = eps
        start_point[3] = eps
        
        res = solve_ivp(self.syst, [0,Max_time],start_point)
        
        return [res.y[0],res.y[1], res.t]

    def paral(self, way):
        x_arr = np.linspace(-np.pi,np.pi,max_el)
        y_arr = np.linspace(-np.pi,np.pi,max_el)
        al_arr = np.linspace(0, np.pi, max_el)
        self.create_path(way)
        self.clean_path(way)

        self.sost = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.calculation)([x, y], al, way) 
                                                    for x in x_arr for y in y_arr for al in al_arr)
        # print(self.sost())
        self.save_fig(self.sost, way)

    def save_fig(self, res, way):
        way = way + "\\"
        for i in range(len(res)):
            plt.plot(res[i][2],np.sin(res[i][0]),label="x")
            plt.plot(res[i][2],np.sin(res[i][1]),label="y", linestyle = '--')
            plt.xlim(0, Max_time)
            plt.ylim(-1.5, 1.5)
            plt.legend()
            plt.savefig(way + f'graph_{i+1}.png')
            # plt.show()
            plt.clf()

    def create_path(self, way):
        if not os.path.exists(way):
            os.makedirs(way)

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
        

if __name__ == "__main__":
    tmp = [5, 1, 1]
    ors = Dinamic(p = tmp)
    # ors.calculation([np.pi, np.pi/2],np.pi/3)
    ors.paral(f"new_life\\res\\n_{ors.N}\\dinamic_sost")
