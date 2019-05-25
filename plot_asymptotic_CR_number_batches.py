import sec_research_py_3
import numpy as np
import quantile_cal
import matplotlib.pyplot as plt
# compare vol
#d=1
xs = [10,20,30,40,50]
#y1 = [4.661, 4.329,4.25,4.218, 4.158]
#y2 = [4.392, 4.146, 4.048,4.003, 3.988]
#d=2
#y1 = [32.4,25.7,23.97,22.96,22.6]
#y2=  [27.8, 22.24,20.91,20.36,20]

#d=5
y1= [46801.7, 8658.06,6013.99,5029.29,4454.77]
y2 =[30547.6, 5366.6,3697.13,3136.98, 2899.14]




#plt.subplot(2, 1, 1)
plt.plot(xs, y1, 'ro--', markersize = 12,label= "Modified Chen's")
plt.plot(xs, y2, 'b>--', markersize = 12, label = "Even")


plt.xlabel("number of batches", fontsize = 20)
#plt.ylabel("CR overall coverage")
plt.ylabel("$v_{d}(m, w)$", fontsize = 20)
plt.legend(loc='upper right', shadow=True, fontsize='xx-large')
#plt.title("d=5")
plt.show()
exit()