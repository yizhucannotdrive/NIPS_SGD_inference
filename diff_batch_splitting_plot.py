import sec_research_py_3
import numpy as np
import quantile_cal
import matplotlib.pyplot as plt
# prelimit performance comparison for three different batch size splitting strategy
#\sigma =0.7
#xs = [150000, 250000, 350000, 450000, 550000, 650000, 750000, 850000, 950000, 1050000]
#y1 = [0.813, 0.851, 0.875, 0.883, 0.888, 0.893, 0.898, 0.917, 0.91, 0.901]
#y2 = [0.711, 0.774, 0.809, 0.82, 0.834, 0.848,0.878, 0.877, 0.877, 0.89]
#y3 = [0.467, 0.531, 0.562, 0.591, 0.643, 0.609, 0.653, 0.66, 0.669, 0.669]
model = "linear"
#sigma =1, logistic regression
if model  == "logistic":
    xs = [100000, 400000, 700000, 1000000]
    y1 = [0.919, 0.942, 0.936, 0.945]
    y2 = [0.867, 0.897, 0.936, 0.94]
    y3 = [0.686, 0.774, 0.826, 0.852]
    CI_1 = [0.017, 0.013,0.015, 0.014]
    CI_2 = [0.021, 0.018, 0.016, 0.018]
    CI_3 = [0.028,0.026, 0.023, 0.02]

#sigma =1, linear regression
if model  == "linear":
    xs = [10000, 40000, 70000, 100000]
    y1 = [0.975, 0.955, 0.97, 0.971]
    y2 = [0.938, 0.947, 0.951, 0.95]
    y3 = [0.787, 0.878, 0.909, 0.912]
    CI_1 = [0.009, 0.013, 0.01, 0.009]
    CI_2 = [0.015, 0.014, 0.013, 0.013]
    CI_3 = [0.025, 0.02,0.017, 0.019]
# naive chenxi's coverage
#y5 = [0.89, 0.919, 0.897, 0.899]

#each dim coverage average

#xs = [100000, 400000, 700000, 1000000]
#y1 = [0.975, 0.982, 0.977, 0.976]
#y2 = [0.945, 0.963, 0.976, 0.978]
#y3 = [0.863, 0.914, 0.93, 0.936]
#y4 = [0.958, 0.978, 0.977, 0.98]
# naive chenxi's coverage
#y5 = [0.905, 0.92, 0.927, 0.932]


# compare vol
#xs = [100000, 400000, 700000, 1000000]
#y1 = [0.001, 0.00029, 0.00017, 0.0001]
#y2 = [0.0007, 0.0002, 0.0001, 0.0001]
#y3 = [0.003, 0.0001, 0.000099, 0.000087]
#y4 = [0.0008, 0.0002, 0.0001, 0.0001]
# naive chenxi's coverage
#y5 = [0.0008, 0.0002, 0.0001, 0.0001]

#plt.subplot(2, 1, 1)
plt.plot(xs, y1, 'ro--', markersize = 6,label= "Modified Chen's")
plt.fill_between(xs, np.subtract(y1, CI_1), np.add(y1, CI_1),  color = 'r', alpha =0.4 )
plt.plot(xs, y2, 'b>--', markersize = 6, label = "Even")
plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2),  color = 'b', alpha =0.4 )
plt.plot(xs, y3, 'gx--', markersize = 6, label = "Decreasing")
plt.fill_between(xs, np.subtract(y3, CI_3), np.add(y3, CI_3),  color = 'g', alpha =0.4)

plt.axhline(y=0.95)
plt.xlabel("total number of iterates")
#plt.ylabel("CR overall coverage")
plt.ylabel("Confidence Region coverage")
plt.legend(loc='lower right', shadow=True, fontsize='x-small')
plt.show()
exit()






