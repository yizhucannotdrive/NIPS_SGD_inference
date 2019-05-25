import matplotlib.pyplot as plt
import numpy as np
d = 20
each_dim_bool  =  True
bool_with_CI = False
xs = [100000, 400000, 700000, 1000000]
if d  == 2 and (not each_dim_bool):
    y1 = [0.919, 0.942, 0.936, 0.945]
    y2 = [0.867, 0.897, 0.936, 0.94]
    y3 = [0.686, 0.774, 0.826, 0.852]
    y4 = [0.89, 0.919, 0.897, 0.899]
    y5 = [0.895, 0.937, 0.938, 0.951]
    y6 = [0.772, 0.873, 0.904, 0.895]
    CI_1 = [0.017, 0.013,0.015, 0.014]
    CI_2 = [0.021, 0.018, 0.016, 0.018]
    CI_3 = [0.028,0.026, 0.023, 0.02]
    CI_4 = [0.019, 0.017, 0.018, 0.018]
    CI_5 = [0.019, 0.015, 0.015, 0.013]
    CI_6 = [0.027, 0.02, 0.018, 0.019]

if d  == 20 and (not each_dim_bool):
    y1 = [0.638, 0.847, 0.878, 0.9]
    y2 = [0.486, 0.77, 0.83, 0.863]
    y3 = [0.171, 0.534, 0.655, 0.733]
    y4 = [0.399, 0.539, 0.549, 0.551]
    y5 = [0.165, 0.57, 0.665, 0.72]
    y6 = [0.036, 0.329, 0.425, 0.518]
    CI_1 = [0.029, 0.02, 0.02, 0.018]
    CI_2 = [0.031, 0.028, 0.023, 0.021]
    CI_3 = [0.023, 0.03, 0.029, 0.027]
    CI_4 = [0.03, 0.03, 0.03, 0.03]
    CI_5 = [0.02, 0.03, 0.029, 0.028]
    CI_6 = [0.015, 0.029, 0.03, 0.03]

if d  == 2 and ( each_dim_bool):
    y1 = [0.9755, 0.982, 0.9775, 0.976]
    y2 = [0.9455, 0.963, 0.976, 0.978]
    y3 = [0.863, 0.914, 0.93, 0.936]
    y4 = [0.905, 0.92, 0.927, 0.932]
    y5 = [0.966, 0.978, 0.982, 0.986]
    y6 = [0.906, 0.953, 0.965, 0.96]
    CI_1 = [0.0096, 0.0082, 0.0092, 0.0094]
    CI_2 = [0.014, 0.012, 0.0095, 0.009]
    CI_3 = [0.021, 0.017, 0.016, 0.015]
    CI_4 = [0.018, 0.017, 0.016, 0.015]
    CI_5 = [0.011, 0.009, 0.008, 0.007]
    CI_6 = [0.018, 0.013, 0.011, 0.012]

if d  == 20 and (each_dim_bool):
    y1 = [0.998, 0.999, 0.999, 0.999]
    y2 = [0.996, 0.999, 0.997, 0.999]
    y3 = [0.982, 0.997, 0.998, 0.998]
    y4 = [0.773, 0.816, 0.83, 0.837]
    y5 = [0.977, 0.996, 0.998, 0.9978]
    y6 = [0.929, 0.988, 0.9927, 0.9943]
    CI_1 = [0.0024, 0.001, 0.001, 0.0009]
    CI_2 = [0.004, 0.001, 0.001, 0.0009]
    CI_3 = [0.008, 0.003, 0.0024, 0.002]
    CI_4 = [0.026, 0.024, 0.023, 0.03]
    CI_5 = [0.009, 0.0035, 0.0026, 0.0028]
    CI_6 = [0.016, 0.0066, 0.005, 0.004]
if bool_with_CI:
    plt.plot(xs, y1, 'ro--', markersize = 6,label= "Modified Chen's")
    plt.fill_between(xs, np.subtract(y1, CI_1), np.add(y1, CI_1),  color = 'r', alpha =0.4 )
    plt.plot(xs, y2, 'b>--', markersize = 6, label = "Even")
    plt.fill_between(xs, np.subtract(y2, CI_2), np.add(y2, CI_2),  color = 'b', alpha =0.4 )
    plt.plot(xs, y3, 'gx--', markersize = 6, label = "Decreasing")
    plt.fill_between(xs, np.subtract(y3, CI_3), np.add(y3, CI_3),  color = 'g', alpha =0.4)
    plt.plot(xs, y4, 'cs--', markersize = 6,label= "Naive Chen's")
    plt.fill_between(xs, np.subtract(y4, CI_4), np.add(y4, CI_4),  color = 'c', alpha =0.4 )
    plt.plot(xs, y5, 'y<--', markersize = 6, label = "HiGrad")
    plt.fill_between(xs, np.subtract(y5, CI_5), np.add(y5, CI_5),  color = 'y', alpha =0.4 )
    plt.plot(xs, y6, 'mv--', markersize = 6, label = "Sectioning")
    plt.fill_between(xs, np.subtract(y6, CI_6), np.add(y6, CI_6),  color = 'm', alpha =0.4)
else:
    plt.plot(xs, y1, 'ro--', markersize=6, label="Modified Chen's")
    plt.plot(xs, y2, 'b>--', markersize=6, label="Even")
    plt.plot(xs, y3, 'gx--', markersize=6, label="Decreasing")
    plt.plot(xs, y4, 'cs--', markersize=6, label="Naive Chen's")
    plt.plot(xs, y5, 'y<--', markersize=6, label="HiGrad")
    plt.plot(xs, y6, 'mv--', markersize=6, label="Sectioning")

plt.axhline(y=0.95)
plt.xlabel("total number of iterates")
# plt.ylabel("CR overall coverage")
if each_dim_bool:
    plt.ylabel("Confidence Interval average coverage")
else:
    plt.ylabel("Confidence Region coverage")

plt.legend(loc='lower right', shadow=True, fontsize='x-small')
plt.show()









