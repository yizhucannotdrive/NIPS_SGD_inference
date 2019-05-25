import sec_research_py_3
import numpy as np
import math
import asymptotic_CI_length_different_batch_size as q_cal
import pickle

r = 0.5
m = 40
TOTAL_num_samples = 10000000

def update_CI_cover_bool(theta_0, hyper_par, r, a, m,c_vec, quantile, quantile_dim, update_mode = "sgd", model = "linear", burn = 0):
    current_theta =  np.copy(theta_0)
    current_theta_sum = np.copy(current_theta)
    #print(current_theta, current_theta_sum)
    current_index =0
    batch_means = []
    d = len(theta_0)
    qd = math.pow(math.pi, float(d) / 2) / math.gamma(float(d) / 2 + 1)
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
    #print(current_theta_sum)
    #print(np.sum(c_vec))
    #print(point_est)
    h_mat = np.matrix(np.zeros(shape=(d, d)))
    for i in range(m):
        frac_scale = np.matrix(batch_means[i] - point_est)
        h_mat += frac_scale.transpose() * frac_scale
    h_mat = h_mat / (m * (m - 1))
    vol = math.sqrt(np.linalg.det(h_mat)) * qd * np.power(quantile, float(d) / 2)
    stat = np.matrix(point_est - hyper_par) * np.linalg.inv(h_mat) * np.matrix(point_est - hyper_par).transpose()
    #print (h_mat)
    #print (stat)
    if stat < quantile:
        cover_bool = 1
    else:
        cover_bool = 0

    cover_bools_each_dim= []
    """
        for i in range(d):
        point_est_dim =  np.copy(point_est)
        point_est_dim[i] = np.copy(hyper_par[i])
        stat_dim = np.matrix(point_est - point_est_dim) * np.linalg.inv(h_mat) * np.matrix(point_est - point_est_dim).transpose()
        if stat_dim < quantile:
            cover_bools_each_dim.append(1)
        else:
            cover_bools_each_dim.append(0)
    """
    for i in range(d):
        point_est_dim = np.copy(point_est[i])
        h_mat_dim = np.copy(h_mat.item((i,i)))
        stat_dim = (point_est_dim - hyper_par[i]) **2 / h_mat_dim
        if stat_dim < quantile_dim:
            cover_bools_each_dim.append(1)
        else:
            cover_bools_each_dim.append(0)
    return cover_bool, vol, cover_bools_each_dim






def main():
    quantile_table = pickle.load(open("quantile_table", "rb"))

    #model = "linear"
    model = "logistic"
    #theta_0 = np.array([0.4])
    #hyper_par = np.array([0.5])
    #theta_0 = np.array([0.23,0.76])
    #hyper_par = np.array([0.33,0.66])
    #theta_0 = np.array([0.35, 0.4, 0.85])
    #hyper_par = np.array([0.25, 0.5, 0.75])
    #high dim

    dim = 10
    hyper_par = np.random.permutation(np.arange(0, 1, 1. / dim))
    print(hyper_par)
    theta_0 = hyper_par + np.random.normal(0, 0.1, dim)

    dim = len(theta_0)
    a = 1
    #update_mode = "iid_gaussian"
    update_mode = "sgd"
    print("d is {}, total num is {},m is {},  model is {}".format(dim, TOTAL_num_samples, m, model))
    burn = 0
    cvecs =[]
    #c_vec1 = []
    #for i in range(m):
    #    c_tmp = int(np.power((i + 1), r / (1 - r)) * np.power(N, 1 / (1 - r)))
    #    c_vec1.append(c_tmp)
    #c_vec1 = (np.array(c_vec1) * (TOTAL_num_samples - burn) / np.float(np.sum(c_vec1))).astype(int)
    N = np.power(TOTAL_num_samples-burn, 1 - r) / m
    eks_1 = (np.array(range(m)) + np.ones(m)) * N
    e_ks = np.power(eks_1, 1 / (1 - r))
    nks = []
    nks.append(int(e_ks[0]))
    for i in range(len(e_ks) - 1):
        nks.append(int(e_ks[i + 1] - e_ks[i]))
    c_vec1 =  nks
    c_vec2 = np.ones(m, dtype=np.int )* int(np.sum(c_vec1) / m)
    c_vec1_rev = np.flip(np.copy(c_vec1))
    print(c_vec1)
    print(c_vec2)
    print(c_vec1_rev)
    """
    cvec_exp = []
    ek = 1
    rate = 1.05
    print("exponential rate is {}".format(rate))
    for i in range(m):
        new_ek = ek * rate + 1
        size =  new_ek - ek
        cvec_exp.append(size)
        ek = new_ek
    cvec_exp = (np.array(cvec_exp) * TOTAL_num_samples/ np.float(np.sum(cvec_exp))).astype(int)
    print(cvec_exp)
    cvec_exp_naive = []
    rate_naive = np.power(TOTAL_num_samples, 1./m)
    cvec_exp_naive.append(rate_naive)
    print(rate_naive)
    cumsum = rate_naive
    for i in range(m-1):
        newcumsum =  cumsum * rate_naive
        val =  newcumsum - cumsum
        cvec_exp_naive.append(val)
        cumsum = np.copy(newcumsum)
    print(cvec_exp_naive)
    """






    # only calculate exponential splitting strategy
    #cvecs.append(cvec_exp)
    # only calculate chen xi's splitting strategy
    #print("only chen xi's")
    cvecs.append(c_vec1)

    # calulate three splitting strategy
    #cvecs.append(c_vec1)
    #cvecs.append(c_vec2)
    #cvecs.append(c_vec1_rev)
    #cvecs.append(cvec_exp)


    num_c_vec3s = 0
    for i in range(num_c_vec3s):
        num_c_vec3_tmp =  np.random.random(m)
        num_c_vec3_tmp = (num_c_vec3_tmp/np.sum(num_c_vec3_tmp) * np.sum(c_vec1)).astype(int)
        print(num_c_vec3_tmp)
        cvecs.append(num_c_vec3_tmp)
    qs =[]
    qs_dim = []
    # only m=30, d=2
    #qs.append(3.54)
    #qs_dim.append(1.91)
    for c_vec_tmp in cvecs:
        c_vec_tmp = np.array(c_vec_tmp)/np.float(np.sum(c_vec_tmp))
        qs.append(q_cal.quantile_limit_dist(c_vec_tmp, m=m, d = dim))
        qs_dim.append(q_cal.quantile_limit_dist(c_vec_tmp, m=m, d=1))
        print(qs_dim)

    #m=10:
    #q1 = 7.38
    #q2 = 10.05
    #m=20:
    #q1 =4.75
    #q2 =7.49
    #m=30:
    #q1 = 4.02
    #q2 = 6.96

    num_rep = 1000
    ss = np.zeros(len(qs))
    #s1 = 0.
    #s2 = 0.
    vols = []
    ss_each_dim  = []
    for i in range(len(qs)):
        vols.append([])
        ss_each_dim .append(np.zeros(dim))
    for i in range(num_rep):
        if i%(num_rep/10) == 0:
            print(i)
        for j in range(len(qs)):
            cover_boolj, vol_j, cover_boolj_each_dim = update_CI_cover_bool(theta_0, hyper_par, r, a, m, cvecs[j], qs[j],qs_dim[j], update_mode = update_mode, model=model, burn = burn)
            #print(cover_boolj_each_dim)
            ss[j] += cover_boolj
            vols[j].append(round(float(vol_j),4))
            ss_each_dim[j] += np.array(cover_boolj_each_dim)


    covs = []
    ave_vols =[]
    for j in range(len(qs)):
        covj = ss[j]/num_rep
        covj_CI_len = 1.96 * np.sqrt(covj * (1 - covj)/ num_rep)
        ave_volj = np.mean(vols[j])
        std_volj =  np.std(vols[j])
        volj_CI_len = 1.96 * std_volj / np.sqrt(num_rep)
        covj_each_dim = ss_each_dim[j] / num_rep
        covj_each_dim_CI_len = 1.96 * np.sqrt(covj_each_dim * (np.ones(dim) - covj_each_dim) / num_rep)
        covj_each_dim_ave = np.mean(covj_each_dim)
        covj_each_dim_ave_CI_len = 1.96 * np.sqrt(covj_each_dim_ave * (1 - covj_each_dim_ave)/ num_rep)
        #covs.append(covj)
        #ave_vols.append(ave_volj)
        print("batch splitting is {}".format(cvecs[j]))
        print("coverage  is {} with CI length {}".format(covj, covj_CI_len))
        print("coverage for each dim is {} with CI length {}".format(covj_each_dim, covj_each_dim_CI_len))
        print("coverage for each dim taking average is {} with CI length {}".format(covj_each_dim_ave, covj_each_dim_ave_CI_len))
        print("averagea CR volume is {} with CI length {} ".format(ave_volj, volj_CI_len))

    exit()

    print("update mode is {}".format(update_mode))
    print("coverage for two batch mean CI xi vs even: {} vs {}".format(covs[0], covs[1]))
    print("Confidence region mean volume for two batch mean CI xi vs even: {} vs {}".format(ave_vols[0], ave_vols[1]))






if __name__ == "__main__":
    main()