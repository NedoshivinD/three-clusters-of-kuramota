import numpy as np
from scipy.integrate import solve_ivp
import joblib 
import matplotlib.pyplot as plt
import os, shutil, sys
from scipy import integrate

Max_time = 100
max_el = 50
max_k = 1
N_JOB = 4
eps = 0.001
speed = 0.01
#4.459709, 2.636232

class Dinamic(object):
    
    #инициализация системы
    def __init__(self,p = [3,1, 1]):
        self.N, self.m, self.omega = p
        self.M = 1
        self.K = 2
        self.alpha = 0
        self.sost = []
        self.ust = []
        self.un_ust = []
        self.t = np.linspace(0,100,Max_time)
        self.k = 1
        self.x_arr = np.linspace(-np.pi,np.pi,max_el)
        self.y_arr = np.linspace(-np.pi,np.pi,max_el)
        self.al_arr = np.linspace(0, np.pi, max_el//2)
        self.k_arr = np.linspace(1,max_k, max_k)
        self.params = []

    def syst(self, t, y0):
        N = self.N
        M = self.M
        K = self.K
        alpha = self.alpha
        m = self.m
        k = self.k
        
        x,y,v,w = y0 #[x,y,w,v] - точки
        f = np.zeros(4)
        # x with 2 dots
        f[0] = v
        # y with 2 dots
        f[1] = w
        # x with 1 dot
        f[2] = 1/m*(k/N * ((M - K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(x-alpha)-(N-M-K)*np.sin(y+alpha)-(N-M-K)*np.sin(x-y-alpha)) - v)
        # y with 1 dot
        f[3] = 1/m*(k/N * ((N-M-2*K)*np.sin(alpha) - M*np.sin(x+alpha) - K*np.sin(y-alpha)-(N-M-K)*np.sin(y+alpha)-M*np.sin(y-x-alpha)) - w)
        
        return f
     
    def calculation(self, x, par):
        M = self.M
        K = self.K
        self.alpha = par[0]
        self.k = par[1]
        # way = way + "\\"
        way = f"new_life\\res\\n_{ors.N}\\dinamic_sost\\"
        start_point=np.zeros(4)
        start_point[0] = x[0] + eps
        start_point[1] = x[1] + eps
        start_point[2] = speed
        start_point[3] = speed 
        res = solve_ivp(self.syst, [0,100], start_point, max_step = 0.1, rtol = 1e-7, atol = 1e-7)
        # self.save_fig_2([res.y[0],res.y[1], res.t],way)
        # params.append([round(x[0],5), round(x[1],5), self.K, self.M, round(self.alpha), self.k])
        return [res.y[0],res.y[1], res.t, self.alpha,self.k]

    def paral(self, way):
        
        self.create_path(way)
        self.clean_path(way)

        self.sost = joblib.Parallel(n_jobs = N_JOB)(joblib.delayed(self.calculation)([x, y], [al,k]) 
                                                    for x in self.x_arr for y in self.y_arr for al in self.al_arr for k in self.k_arr)
        # print(self.sost())
        self.save_fig(self.sost, way)
        # print("kek")

    def save_fig(self, res, way):
        way = way + "\\"
        for i in range(len(res)):
            param = [np.round(res[i][0][0],5), np.round(res[i][1][0],5), self.K, self.M, res[i][3],res[i][4]]
            plt.plot(res[i][2],np.angle(np.exp(1j*res[i][0])),label="x")
            plt.plot(res[i][2],np.angle(np.exp(1j*res[i][1])),label="y", linestyle = '--')
            plt.xlim(0, 100)
            plt.ylim(-np.pi-0.2, np.pi+0.2)
            plt.text(x=50,y=np.pi+0.3, horizontalalignment = 'center', s="x0 = " + str(param[0]) + ", y0 = " + str(param[1])
                    +  ", K = " + str(param[2]) + ", M = " + str(param[3]) + ", alpha = " + str(param[4])
                     + ", k = " + str(param[5]))
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
    way = f"new_life\\res\\n_{ors.N}\\dinamic_sost"
    ors.paral(way)
    
    # res = ors.calculation([-3.14059, -3.14259],[3.141592653589793,1])
    # plt.plot(res[2],np.angle(np.exp(1j*res[0])),label="x")
    # plt.plot(res[2],np.angle(np.exp(1j*res[1])),label="y", linestyle = '--')
    # plt.xlim(0, 100)
    # plt.ylim(-np.pi-0.2, np.pi+0.2)
    # par = [-3.14059, -3.14259, 2, 1, 3.141592653589793, 1.0]
    # plt.text(x=50,y=np.pi+0.3, horizontalalignment = 'center', s="x = " + str(par[0]) + ", y = " + str(par[1]) +  ", K = " + str(par[2]) + ", M = " + str(par[3]) + ", alpha = " + str(par[4]) + ", k = " + str(par[5]))
    # plt.legend()
    # plt.show()
    
