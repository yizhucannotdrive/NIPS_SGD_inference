import sec_research_py_3
import numpy as np
import quantile_cal
import matplotlib.pyplot as plt

xs = [100000, 400000, 700000, 1000000]
y1 = [0.889, 0.905, 0.943, 0.942]
y2 = [0.917, 0.933, 0.928, 0.934]
y3 = [0.903, 0.921, 0.922, 0.92]
y4 = [0.409, 0.689, 0.764, 0.801]
CI_1 = [0.019, 0.018,0.014, 0.014]
CI_2 = [0.017, 0.0155, 0.016, 0.015]
CI_3 = [0.018,0.017, 0.016, 0.017]
CI_4 = [0.03, 0.028, 0.026, 0.025]
# naive chenxi's coverage

plt.plot(xs, y1, 'ro--', markersize = 6,label= "Exponential(1.01)")
plt.fill_between(xs, np.subtract(y1, CI_1), np.add(y1, CI_1),  color = 'r', alpha =0.4 )
plt.plot(xs, y2, 'b>--', markersize = 6, label = "Exponential(1.05)")
plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2),  color = 'b', alpha =0.4 )
plt.plot(xs, y3, 'gx--', markersize = 6, label = "Exponential(1.1)")
plt.fill_between(xs, np.subtract(y3, CI_3), np.add(y3, CI_3),  color = 'g', alpha =0.4)
plt.plot(xs, y4, 'ys--', markersize = 6, label = "Exponential(1.4)")
plt.fill_between(xs, np.subtract(y4, CI_4), np.add(y4, CI_4), color = 'y', alpha =0.4)


plt.axhline(y=0.95)
plt.xlabel("total number of iterates")
plt.ylabel("CR  coverage")
plt.legend(loc='lower right', shadow=True, fontsize='x-small')
plt.show()
exit()