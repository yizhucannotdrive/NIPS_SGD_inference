import asymptotic_CI_length_different_batch_size as q_cal
import numpy as np
import scipy.stats

def main():
    ms = [120,150]
    ds = [100]
    r = 0.5
    TOTAL_num_samples = 100000
    burn = 0
    num_rep = 10
    for m in ms:
        for d in ds:
            print("m is {}, d is {}".format(m,d))
            N = np.power(TOTAL_num_samples -burn, 1-r)/m
            eks_1 = (np.array(range(m)) + np.ones(m))*N
            e_ks = np.power(eks_1, 1/(1-r))
            nks = []
            nks.append(int(e_ks[0]))
            for i in range(len(e_ks)-1):
                nks.append(int(e_ks[i+1] -e_ks[i] ))
            c_vec2 = np.ones(m, dtype=np.int) * int(np.sum(nks) / m)
            #print(nks)
            #print(c_vec2)
            nks = np.array(nks) / np.float(np.sum(nks))
            c_vec2 = np.array(c_vec2) / np.float(np.sum(c_vec2))
            #print(nks)

            quantiles = []
            for i in range(num_rep):
                quantile_sample = q_cal.quantile_limit_dist(nks, m=m, d=d)
                #quantile_sample = q_cal.quantile_limit_dist(c_vec2, m=m, d=d)/m
                quantiles.append(quantile_sample)
            ave_quan = np.mean(quantiles)
            std_quan = np.std(quantiles)
            ave_quan_CI_len = 1.96 * std_quan / np.sqrt(num_rep)
            print("nks")
            print("m is {}, d is {}, mean is {}, CI len is {}".format(m,d, ave_quan, ave_quan_CI_len))

if __name__ == "__main__":
    main()