import numpy as np
from matplotlib import pyplot as plt

def w_j_t_cal(j, t, A = 2):
    s=1
    for i in range(j+1,t):
        prod = 1
        for k in range(j+1, i+1):
            prod *= (1-A*np.power(k, -0.95))
        s += prod
    s *= np.power(j, -0.95)
    s = s- 1./A
    return s

def main():
    t_vec = [10,100,1000,10000, 100000]
    for t in t_vec:
        vals = []
        stat = 0
        A = 2.0
        var_Stat = 0.0
        for j in range(1, t):
            #if j%(t/10)==0:
            #    print(j)
            val_tmp = w_j_t_cal(j,t, A = A)
            #print("w_{}_{} is {}".format(j,t, val_tmp))
            vals.append(val_tmp)
            stat +=  val_tmp ** 2 + 2* val_tmp / A
            var_Stat +=  val_tmp ** 2
        var_Stat /= t
        print("current t is {}, var_stat is {}".format(t, var_Stat))
        """
        plt.hist(np.array(vals), color='blue', edgecolor='black',
                bins=int(10))
        plt.show()
        stat = A**2 * stat/t
        print(" t is {}, Stat is {}".format(t, stat))
        
        """



if __name__ == "__main__":
    main()
