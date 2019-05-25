import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

alpha = 0.95
a = 0.5


def h_B_func(c_vec,m, d):
    h_mat = np.matrix(np.zeros(shape=(d,d)))
    bm_frac = []
    bm_one = np.zeros(d)
    for i in range(m):
        frac_tmp = np.random.normal(0,np.sqrt(c_vec[i]), d)
        bm_frac.append(frac_tmp)
        bm_one += frac_tmp
    for i in range(m):
        frac_scale = np.matrix(bm_frac[i] / c_vec[i] - bm_one)
        h_mat += frac_scale.transpose() * frac_scale
    h_mat =  h_mat/(m*(m-1))
    return h_mat, bm_one

def quantile_limit_dist(c_vec, m, d,  numsamples = 100000):
    samples =[]
    for i in range(numsamples):
        h_mat, bm_one = h_B_func(c_vec,m=m, d=d)
        sample = np.matrix(bm_one) * np.linalg.inv(h_mat)*np.matrix(bm_one).transpose()
        samples.append(sample)
    quantile = np.percentile(samples, alpha * 100)
    #print(quantile)
    return quantile

def expec_vol_h_B_func(c_vec, numsamples, quantile, m, d):
    qd = math.pow(math.pi, float(d) / 2) / math.gamma(float(d) / 2 + 1)
    samples = []
    for i in range(numsamples):
        mat, _ = h_B_func(c_vec, m=m, d=d )
        vol_sample = math.sqrt(np.linalg.det(mat)) * qd * np.power( quantile ,float(d)/2)
        #old version, not right
        #vol_sample = math.sqrt(np.linalg.det(mat)) * qd * np.power(quantile, float(d) / 2) * np.power(1./m, float(d)/2)
        samples.append(vol_sample)
    vol_mean =  np.mean(samples)
    print(vol_mean)
    return vol_mean


def cal_cvec_kapa(kappa, m):
    cvec = []
    ci = 1
    for i in range(m):
        cvec.append(ci)
        ci *= np.exp(kappa)
    csum = np.sum(cvec)
    cvec = np.array(cvec)/csum
    return cvec


def main():
    # m = 3
    m = 30
    d = 2
    num_samples = 100000
    # each c_k before normalization is k^(a/(1-a)) * N^(1/(1-a))
    # compare different batch number for chenxi's splitting and even splitting
    """
    print("d is {}, m is {}".format(d,m))
    c_vec1 = []
    for i in range(m):
        c_tmp = np.power((i+1), a/(1-a)) * np.power(N, 1/(1-a))
        c_vec1.append(c_tmp)
    #print(np.sum(c_vec1))
    c_vec1 =  np.array(c_vec1)/ np.sum(c_vec1)
    c_vec2 =  np.ones(m)/m
    print(c_vec1)
    lim_quantile_1 = quantile_limit_dist(c_vec1, num_samples, d=d)
    lim_quantile_2 = quantile_limit_dist(c_vec2, num_samples, d=d)
    CI_vol_1 = expec_vol_h_B_func(c_vec1, num_samples, lim_quantile_1)
    CI_vol_2 = expec_vol_h_B_func(c_vec2, num_samples, lim_quantile_2)
    print(" CI vol for Xi's batch size allocation vs equal batch size allocation are {} and {}". format(CI_vol_1, CI_vol_2))
    exit()



    #kapas = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]
    print("dim is {}".format(d))
    kapas = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    cvols = []
    quantiles = []
    for kappa in kapas:
        cvec =   cal_cvec_kapa(kappa, m)
        print(cvec)
        lim_quantile = quantile_limit_dist(cvec, num_samples)
        quantiles.append(lim_quantile)
        CI_vol = expec_vol_h_B_func(cvec, num_samples, lim_quantile)
        cvols.append(CI_vol)
    print(cvols)
    quantiles = np.power(np.array(quantiles), float(d)/2)
    print(quantiles)
    meansample_variance = np.array(cvols) / quantiles
    print(meansample_variance)
    plt.plot(kapas, cvols, 'ro')
    plt.ylim((0, np.mean(cvols) * 3))
    plt.show()

    plt.plot(kapas, quantiles, 'ro')
    plt.ylim((0, np.mean(quantiles) * 3))
    plt.show()

    plt.plot(kapas, meansample_variance, 'ro')
    plt.ylim((0, np.mean(meansample_variance) * 3))
    plt.show()
    exit()

    #check convexity of CI length
    cvols = []
    c1 = 0.1
    c1s = []
    quantiles = []
    while c1<0.999:
        c1s.append(c1)
        c2 = 1 - c1
        cvec = np.array([c1,c2])
        print(cvec)
        lim_quantile = quantile_limit_dist(cvec, num_samples)
        quantiles.append(lim_quantile)
        CI_vol = expec_vol_h_B_func(cvec, num_samples, lim_quantile)
        cvols.append(CI_vol)
        c1 += 0.1

    print(cvols)
    quantiles = np.sqrt(np.array(quantiles))
    print(quantiles)
    meansample_variance =  np.array(cvols)/ quantiles
    print(meansample_variance)
    plt.plot(c1s, cvols, 'ro')
    plt.ylim((0, np.mean(cvols)*3))
    plt.show()

    plt.plot(c1s, quantiles, 'ro')
    plt.ylim((0, np.mean(quantiles)*3))
    plt.show()

    plt.plot(c1s, meansample_variance, 'ro')
    plt.ylim((0, np.mean(meansample_variance)*3))
    plt.show()
    exit()
    ##two batch CI length, quantiles, sample variance mean
    ##[20.270613470699054, 20.19516074353915, 20.29642967795582, 20.45396270200075, 20.515064485113808, 20.280779217679665, 20.195384295890335, 20.26900521899515, 20.37116514936713]
    ##[35.38050236679671, 75.24807377695916, 117.3374242345477, 151.7710171894241, 165.43665939097193, 149.65930275439507, 116.07023036345413, 75.95206724645682, 35.88375762578609]
    ##[0.57293176 0.2683811  0.1729749  0.13476857 0.12400555 0.13551299, 0.1739928  0.26686575 0.56769877]
    
"""
    cvols = []
    c1 = 0.1
    c2 = 0.1
    c1s = []
    c2s = []
    quantiles = []
    while c1<0.999:
        while c2< 1-c1-0.01:
            c1s.append(c1)
            c2s.append(c2)
            c3 = 1 - c1-c2
            cvec = np.array([c1,c2,c3])
            print(cvec)
            lim_quantile = quantile_limit_dist(cvec, num_samples, m=m, d=d)
            quantiles.append(lim_quantile)
            CI_vol = expec_vol_h_B_func(cvec, num_samples, lim_quantile,m=m ,d=d)
            cvols.append(CI_vol)
            c2 += 0.1
        c1 += 0.1
        c2 = 0.1
    print(cvols)
    print(quantiles)
    c1s = np.array(c1s)
    c2s = np.array(c2s)
    cvols = np.array(cvols)
    quantiles = np.sqrt(np.array(quantiles))
    meansample_variance =  np.array(cvols)/ quantiles
    print(meansample_variance)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.scatter(c1s, c2s, cvols, c='r', marker='o')
    ax.set_zlim(0, np.mean(cvols)*2)
    #ax.contour3D(c1s, c2s, cvols,  50, cmap='binary')
    plt.show()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.scatter(c1s, c2s, quantiles, c='r', marker='o')
    ax.set_zlim(0, np.mean(quantiles)*2)
    plt.show()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.scatter(c1s, c2s, meansample_variance, c='r', marker='o')
    ax.set_zlim(0, np.mean(meansample_variance)*2)

    plt.show()


if __name__ == "__main__":
    main()