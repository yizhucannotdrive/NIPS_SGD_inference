import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
degenerate = False
if not degenerate:
    dets = pickle.load( open( "det_chen_xi_degenerate", "rb" ) )
    n1, n2, n3 = plt.hist(x = dets, bins = 50,weights=np.ones(len(dets)) / len(dets),  facecolor='blue', edgecolor = 'black', range= (0, 0.6e15))
else:
    dets = np.zeros(100)
    n1, n2, n3 = plt.hist(x = dets, bins = 50,weights=np.ones(len(dets)) / len(dets),  facecolor='blue', edgecolor = 'black')
plt.xlabel("determinant value", fontsize =20)
plt.ylabel("percentage", fontsize =20)
plt.show()
