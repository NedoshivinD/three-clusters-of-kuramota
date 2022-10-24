from re import L
from tkinter import Y
from lib import *
import numpy as np
from numpy import linalg as LA
from Dinamics import *

class Three_klasters(object):
    
    def __init__(self) -> None:
        pass
        
    def matrix(self,param):
        x,y,K,M,alpha = param
        f = []
        f.append([0, 0, 1, 0])
        f.append([0, 0, 0, 1])
        f.append([1/(N*m)*(-M*np.cos(x+alpha) - K*np.cos(x - alpha) - (N - M - K)*np.cos(x-y-alpha)),
                1/(N*m)*(-(N-M-K)*np.cos(y + alpha) + (N-M-K)*np.cos(x-y-alpha)), -1/m, 0])
        f.append([1/(N*m)*(-M*np.cos(x+y) + M*np.cos(y-x-alpha)),
                1/(N*m)*(-M*np.cos(x+y)-K*np.cos(y-alpha)-(N-M-K)*np.cos(y+alpha)-M*np.cos(y-x-alpha)),
                0, -1/m])
        return(f)
    
    def yakobi(self,param):   
        matrix_yak = self.matrix(param= param)
        lam, vect = LA.eig(matrix_yak)
        return(lam)

    def razbor_txt(self,name = "res\\n_\\res_n_.txt", chislo = 9):
        tmp = name[0:-4]+f"{chislo}"+name[-4:]
        tmp = tmp[0:6]+f"{chislo}"+tmp[6:]
        ress = []
        with open(tmp) as file:
            for line in file:
                ress.append(razb_str(line.rstrip()))
        return ress
    
    def some_sob_numb(self,param):# массив вида [[*,*,*,*],[*,*,*,*]...], если элемент 1-н то [[*,*,*,*]]
        z = 0
        for i in param:
            z+=1
            print(f"{z} собственные числа:")
            for j in self.yakobi(param = i):
                print(j)

    def write_stability(self,ress,name = "res\\n_\\stability_sost_ravn.txt",chislo = 9):
        tmp = name[0:6]+f"{chislo}"+name[6:]
        with open(tmp,"w") as file:
            for i in range(len(ress)):
                lam = self.yakobi(param=ress[i]) 
                g = 1
                for j in lam:
                    if j.real < 0:
                        continue
                    else:
                        g = 0
                        break  
                if g == 1:
                    file.write(str(ress[i]) + '\n')

    def write_non_stability(self,ress=[],name="res\\n_\\non_stability_sost_ravn.txt",chislo = 9):
        tmp = name[0:6]+f"{chislo}"+name[6:]
        with open(tmp,"w") as file:
            for i in range(len(ress)):
                lam = self.yakobi(param = ress[i]) 
                g = 1
                for j in lam:
                    if j.real < 0:
                        continue
                    else:
                        g = 0
                        break  
                if g != 1:
                    file.write(str(ress[i]) + '\n')    


#---------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    klast = Three_klasters()
    N=3
    m=1
    tmp = klast.razbor_txt(chislo = 3)  #defolt chislo = 9
   
    # for i in tmp[0:10]:
    #     print(i)

    klast.some_sob_numb(param=[np.pi, np.pi, 1, 1, np.pi])
    
    klast.write_stability(ress= tmp)





