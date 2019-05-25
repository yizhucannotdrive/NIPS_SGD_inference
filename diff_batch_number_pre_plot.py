import sec_research_py_3
import numpy as np
import quantile_cal
import matplotlib.pyplot as plt

d = 1
xs = [10000, 40000, 70000, 100000]
if_CI = True
if d==1:
    y1 = [0.934, 0.932, 0.946, 0.948]
    y2 = [0.899, 0.917, 0.934, 0.933]
    y3 = [0.842, 0.908, 0.932, 0.93]
    y4 = [0.794, 0.889, 0.908, 0.906]

    CI_1 = [0.015, 0.015,0.014, 0.013]
    CI_2 = [0.018, 0.018, 0.015, 0.015]
    CI_3 = [0.023,0.017, 0.018, 0.015]
    CI_4 = [0.025,0.019, 0.018, 0.018]
if d==2:
    y1 = [0.922, 0.943, 0.942, 0.93]
    y2 = [0.845, 0.914, 0.923, 0.927]
    y3 = [0.783, 0.889, 0.916, 0.927]
    y4 = [0.74, 0.86, 0.882, 0.905]

    CI_1 = [0.016, 0.014,0.014, 0.015]
    CI_2 = [0.022, 0.017, 0.016, 0.016]
    CI_3 = [0.025,0.019, 0.017, 0.016]
    CI_4 = [0.027,0.021, 0.019, 0.018]
if d==3:
    y1 = [0.913, 0.933, 0.947, 0.933]
    y2 = [0.814, 0.897, 0.919, 0.927]
    y3 = [0.73, 0.876, 0.909, 0.906]
    y4 = [0.615, 0.817, 0.845, 0.883]

    CI_1 = [0.017, 0.015,0.013, 0.015]
    CI_2 = [0.024, 0.018, 0.017, 0.016]
    CI_3 = [0.027,0.02, 0.017, 0.018]
    CI_4 = [0.03,0.024, 0.022, 0.019]


plt.plot(xs, y1, 'ro--', markersize = 6,label= "m=10")
plt.plot(xs, y2, 'b>--', markersize = 6, label = "m=20")
plt.plot(xs, y3, 'gx--', markersize = 6, label = "m=30")
plt.plot(xs, y4, 'ys--', markersize = 6, label = "m=40")
if if_CI:
    plt.fill_between(xs, np.subtract(y1, CI_1), np.add(y1, CI_1),  color = 'r', alpha =0.4 )
    plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2),  color = 'b', alpha =0.4 )
    plt.fill_between(xs, np.subtract(y3, CI_3), np.add(y3, CI_3),  color = 'g', alpha =0.4)
    plt.fill_between(xs, np.subtract(y4, CI_4), np.add(y4, CI_4), color = 'y', alpha =0.4)

plt.axhline(y=0.95)
plt.xlabel("total number of iterates")
#plt.ylabel("CR overall coverage")
plt.ylabel("Confidence Region coverage")
plt.legend(loc='lower right', shadow=True, fontsize='x-small')
plt.show()
exit()






