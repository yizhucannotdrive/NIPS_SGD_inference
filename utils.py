import numpy as np
from scipy.stats import f
import pickle

def m_times_F_quantiles(m, d, alpha):
    F_quantile = f.ppf(1- alpha, d, m-d)
    m_quantile = F_quantile * d *(m-1)/((m-d)*m)
    return m_quantile
def read_pickle(dir):
    tmp_data = pickle.load(open(dir, "rb"))
    for key in tmp_data.keys():
        print("***************************************************8")
        print(key)
        for name in tmp_data[key].keys():
            print("{} is {}".format(name, tmp_data[key][name]))

def main():
    d = 10
    ms = [15,20,25,30,35]
    alphas = [0.1, 0.05]
    for m in ms:
        for alpha in alphas:
            F_q =  m_times_F_quantiles(m , d, alpha)
            print("alpha is {}, m is {}, d is{}, F quantile is {}".format(alpha, m, d, F_q))
    exit()

    #dir = "./result.p"
    #read_pickle(dir)
    A = np.zeros((m,m))
    d_vec = np.random.random(m)
    d_sum = np.sum(d_vec)
    d_vec = d_vec / d_sum
    print(d_vec)
    for i in range(m):
        for j in range(m):
            if i==j:
                A[i][j] = 1/d_vec[i]-2 +m*d_vec[i]
            else:
                A[i][j] = -np.sqrt(d_vec[i]/d_vec[j])-np.sqrt(d_vec[j]/d_vec[i]) +m * np.sqrt(d_vec[i] * d_vec[j])
    A_truncat = A[1:m,1:m]
    A_truncat_rank = np.linalg.matrix_rank(A)
    rank = np.linalg.matrix_rank(A)
    print("rank of A is {}".format(rank))
    print("rank of A _truncate is {}".format(A_truncat_rank))


if __name__ == "__main__":
    main()