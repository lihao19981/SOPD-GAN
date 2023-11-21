from sko.PSO import PSO
import math
import numpy as np

def Fun(X):
    x ,y,w =X
    var_x = 1
    var_y = 1
    k_theta = 0.1
    k_vel = 0.1
    val_stc = 1
    ff = x ** 2 + (y - 0.05) ** 2 +w ** 2
    f= -math.exp(x ** 2 + (y - 0.05) ** 2 +w ** 2)
    f1 = math.exp(-((x - 1)/var_x) ** 2)
    f2 = math.exp(-((y - 1)/var_y) ** 2)
    f3 = math.exp(k_theta * math.cos(w - 0))
    f4 = math.exp(k_vel * math.cos(math.atan( y / x ) - 1))
    f5 = 1/(1+math.exp(-val_stc * (1 +1)))

    return  -f1  *f2*f3

dim = 3
lb = np.array([-10.0] * dim)

ub = np.array([10.0] * dim)
pso = PSO(func=Fun, dim=3,pop=40, max_iter=100, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
fitness = pso.run()
print('best_x is ',pso.gbest_x)
print('best_y is ',pso.gbest_y)
