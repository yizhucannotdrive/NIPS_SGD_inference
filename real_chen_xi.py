import sec_research_py_3
import numpy as np
import math
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle
r = 0.5
N = 10
TOTAL_num_samples = 1000000
M_rate = 0.25
m = int(np.power(TOTAL_num_samples, M_rate))
print(m)
alpha = 0.95
z_q = scipy.stats.norm.ppf(0.975)

def update_CI_cover_bool(theta_0, hyper_par, r, a, m,c_vec, update_mode = "sgd", model = "linear", z_quantile = z_q, burn = 0):
    current_theta =  np.copy(theta_0)
    current_theta_sum = np.copy(current_theta)
    #print(current_theta, current_theta_sum)
    current_index =0
    batch_means = []
    d = len(theta_0)
    current_theta, current_theta_sum, _, current_index = sec_research_py_3.update_multi(
        current_theta, current_theta_sum, current_index, burn, hyper_par, r, a, update_mode=update_mode,
        current_noise_sum=0, model=model)
    burn_theta_sum = np.copy(current_theta_sum)
    for c in c_vec:
        pre_theta_sum = np.copy(current_theta_sum)
        current_theta, current_theta_sum, _, current_index = sec_research_py_3.update_multi(
            current_theta, current_theta_sum, current_index, c, hyper_par, r, a, update_mode=update_mode,
            current_noise_sum=0, model = model)
        x_tmp_batch_mean = (current_theta_sum - pre_theta_sum)/float(c)
        batch_means.append(x_tmp_batch_mean)
        #print(current_theta_sum)
    point_est = (current_theta_sum - burn_theta_sum) /np.sum(c_vec)
    h_mat = np.matrix(np.zeros(shape=(d, d)))
    for i in range(m):
        frac_scale = np.matrix(batch_means[i] - point_est)
        h_mat += c_vec[i] * frac_scale.transpose() * frac_scale
    h_mat = h_mat / m
    det_h = np.linalg.det(h_mat)
    cover_bool = 1.
    vol = 1
    for i in range(d):
        half_len = z_quantile * np.sqrt(h_mat.item((i,i))) / np.sqrt(TOTAL_num_samples)
        vol = vol *  half_len * 2
        if hyper_par[i] > point_est[i] + half_len or hyper_par[i] < point_est[i] - half_len:
            cover_bool = 0.
    return cover_bool, vol , det_h

def update_CI_cover_bool_each_dim(theta_0, hyper_par, r, a, m,c_vec, update_mode = "sgd", model = "linear", z_quantile = z_q, burn  = 0):
    current_theta =  np.copy(theta_0)
    current_theta_sum = np.copy(current_theta)
    #print(current_theta, current_theta_sum)
    current_index =0
    batch_means = []
    d = len(theta_0)
    current_theta, current_theta_sum, _, current_index = sec_research_py_3.update_multi(
        current_theta, current_theta_sum, current_index, burn, hyper_par, r, a, update_mode=update_mode,
        current_noise_sum=0, model=model)
    burn_theta_sum = np.copy(current_theta_sum)
    for c in c_vec:
        pre_theta_sum = np.copy(current_theta_sum)
        current_theta, current_theta_sum, _, current_index = sec_research_py_3.update_multi(
            current_theta, current_theta_sum, current_index, c, hyper_par, r, a, update_mode=update_mode,
            current_noise_sum=0, model = model)
        x_tmp_batch_mean = (current_theta_sum - pre_theta_sum)/float(c)
        batch_means.append(x_tmp_batch_mean)
        #print(current_theta_sum)
    point_est = (current_theta_sum - burn_theta_sum) /np.sum(c_vec)
    h_mat = np.matrix(np.zeros(shape=(d, d)))
    for i in range(m):
        frac_scale = np.matrix(batch_means[i] - point_est)
        h_mat += c_vec[i] * frac_scale.transpose() * frac_scale
    h_mat = h_mat / m
    det_h = np.linalg.det(h_mat)
    cover_bools= []
    vols = []
    for i in range(d):
        cover_bool = 1.
        half_len = z_quantile * np.sqrt(h_mat.item((i,i))) / np.sqrt(TOTAL_num_samples)
        vols.append(half_len * 2)
        if hyper_par[i] > point_est[i] + half_len or hyper_par[i] < point_est[i] - half_len:
            cover_bool = 0.
        cover_bools.append(cover_bool)
    return cover_bools, vols, det_h


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def main():
    #model = "linear"
    model = "logistic"
    #theta_0 = np.array([0.4])
    #hyper_par = np.array([0.5])
    #theta_0 = np.array([0.23, 0.76])
    #hyper_par = np.array([0.33, 0.66])
    # theta_0 = np.array([0.35, 0.4, 0.85])
    # hyper_par = np.array([0.25, 0.5, 0.75])
    # high dim:
    dim = 20
    hyper_par = np.random.permutation(np.arange(0, 1, 1. / dim))
    print(hyper_par)
    theta_0 = hyper_par + np.random.normal(0, 0.1, dim)

    dim = len(theta_0)
    burn = 0
    z_quantile= scipy.stats.norm.ppf(1-(1-alpha)/(dim*2))
    print("z prob for each dimension is {}".format(1-(1-alpha)/(dim*2)))
    a = 1
    #update_mode = "iid_gaussian"
    update_mode = "sgd"
    print("d is {}, total num is {},m is {},  model is {}".format(dim, TOTAL_num_samples, m, model))

    # coverage and volume for each dim
    #c_vec1 = []
    #for i in range(m):
    #    c_tmp = int(np.power((i + 1), r / (1 - r)) * np.power(N, 1 / (1 - r)))
    #    c_vec1.append(c_tmp)
    #c_vec1 = (np.array(c_vec1) * (TOTAL_num_samples - burn) / np.float(np.sum(c_vec1))).astype(int)
    #print(c_vec1)
    N = np.power(TOTAL_num_samples - burn, 1 - r) / m
    eks_1 = (np.array(range(m)) + np.ones(m)) * N
    e_ks = np.power(eks_1, 1 / (1 - r))
    nks = []
    nks.append(int(e_ks[0]))
    for i in range(len(e_ks) - 1):
        nks.append(int(e_ks[i + 1] - e_ks[i]))
    c_vec1 = nks
    #c_vec_cum_sum = np.cumsum(c_vec1)
    #for i in range(len(c_vec_cum_sum)-1):
    #    print(np.divide(np.float(c_vec_cum_sum[i+1]),c_vec_cum_sum[i]), np.power((np.float(i+2)/(i+1)), 2))
    #exit()

    num_rep = 1000

  
    vols = np.zeros(dim)
    ss = np.zeros(dim)
    allvols = []
    det_hs = []
    for i in range(num_rep):
        if i % (num_rep / 10) == 0:
            print(i)
        cover_bools, vol, det_h = update_CI_cover_bool_each_dim(theta_0, hyper_par, r, a, m, c_vec1, update_mode=update_mode,
                                                         model=model, z_quantile=z_q, burn = burn)
        ss += np.array(cover_bools)
        vols += np.array(vol)
        allvols.append(vol)
        if det_h < 0 or np.abs(det_h) < 0.0000001:
            det_h = 0
        det_hs.append(det_h)
    cov = ss / num_rep
    cov_CI_len =  1.96 * np.sqrt(cov * (np.ones(len(cov)) - cov) / num_rep)
    ave_vol = vols / num_rep
    allvols = np.array(allvols)
    std_vol = np.std( allvols, 0)
    vols_CI_len = 1.96 * std_vol / np.sqrt(num_rep)
    print(" coverage for each dim with alpha = 95%")

    print(" each dim coverage  is {} with CI length {}".format(cov, cov_CI_len))
    # coverage is [0.86 0.87]
    print("each dim averagea CR volume is {} with CI length {}".format(ave_vol, vols_CI_len ))

    ## average across dimension
    overal_cov_dim = np.mean(cov)
    overal_cov_CI_len =  1.96 * np.sqrt(overal_cov_dim * (1 - overal_cov_dim) / num_rep)
    print("coverage for each dim taking average is {} with CI length {}".format(overal_cov_dim,
                                                                                overal_cov_CI_len))
    pickle.dump(det_hs, open("det_chen_xi_degenerate", "wb"))
    #n1, n2, n3 = plt.hist(x = det_hs, bins = [0, 0.000001, 0.000002, 0.000003, 0.000004, 0.000005], normed=1, facecolor='blue', alpha=0.5)
    #plt.show()
    #n1, n2, n3 = plt.hist(x = det_hs, bins = 50, weights=np.ones(len(det_hs)) / len(det_hs), facecolor='blue', edgecolor = 'black')
    #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    #plt.xlim((0, 0.4*10**7))
    #plt.xlabel("determinant value")
    #plt.ylabel("percentage")
    #plt.show()
    #exit()

    """
    vols = np.zeros(dim)
    ss = np.zeros(dim)
    allvols = []
    for i in range(num_rep):
        if i % (num_rep / 10) == 0:
            print(i)
        cover_bools, vol, det_h = update_CI_cover_bool_each_dim(theta_0, hyper_par, r, a, m, c_vec1, update_mode=update_mode,
                                                         model=model, z_quantile=z_quantile)
        ss += np.array(cover_bools)
        vols += np.array(vol)
        allvols.append(vol)
    cov = ss / num_rep
    cov_CI_len = 1.96 * np.sqrt(cov * (np.ones(len(cov)) - cov) / num_rep)
    ave_vol = vols / num_rep
    allvols = np.array(allvols)
    std_vol = np.std(allvols, 0)
    vols_CI_len = 1.96 * std_vol/ np.sqrt(num_rep)
    print(" coverage for each dim with alpha = {} %".format(1-(1-alpha)/dim))

    print(" each dim coverage  is {} with CI length {}".format(cov, cov_CI_len))
    # coverage is [0.86 0.87]
    print("each dim averagea CR volume is {} with CI length {}".format(ave_vol, vols_CI_len))
    ## average across dimension
    overal_cov_dim = np.mean(cov)
    overal_cov_CI_len = 1.96 * np.sqrt(overal_cov_dim * (1 - overal_cov_dim) / num_rep)
    print("coverage for each dim taking average is {} with CI length {}".format(overal_cov_dim,
                                                                                overal_cov_CI_len))
                                                                                
    """


    # overall coverage and volume
    vols = []
    ss = 0.
    for i in range(num_rep):
        if i%(num_rep/10) == 0:
            print(i)
        cover_bool, vol, det_h= update_CI_cover_bool(theta_0, hyper_par, r, a, m, c_vec1, update_mode = update_mode, model=model, z_quantile=z_quantile, burn = burn )
        ss += cover_bool
        vols.append(round(float(vol),4))


    cov = ss/num_rep
    cov_CI_len =  1.96 * np.sqrt(cov * (1 - cov) / num_rep)
    ave_vol = np.mean(vols)
    std_vol  = np.std(vols)
    ave_vol_CI_len = 1.96 * std_vol / np.sqrt(num_rep)
    print("coverage  is {} with CI length {}".format(cov, cov_CI_len))
    print("averagea CR volume is {} with CI length {}".format(ave_vol, ave_vol_CI_len))






if __name__ == "__main__":
    main()