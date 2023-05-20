import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(0,2*np.pi,1000)
x = np.sin(z)
y = np.cos(z)
p1 = 1
p2 = p1 + 4.459709
p3 = p1 + 2.636232
x1 = np.sin(p1)
y1 = np.cos(p1)
x2 = np.sin(p2)
y2 = np.cos(p2)
x3 = np.sin(p3)
y3 = np.cos(p3)
# 2, 1, 2.0944
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(x, y)
plt.scatter(x1,y1,c='r',label='первый кластер',alpha=0.5,marker='v')
plt.scatter(x1,y1,c='r',alpha=0.5,marker='v')
# plt.scatter(1,0,c='r',alpha=0.5,marker='v')

plt.scatter(x2,y2,c='g',label='второй кластер',alpha=0.5,marker='<')

plt.scatter(x3,y3,c='y',label='третий кластер',alpha=0.5)
plt.scatter(x3,y3,c='y',alpha=0.5)


plt.grid()
plt.legend()
ax.set_aspect('equal', adjustable='box')
plt.show()
