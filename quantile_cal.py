import numpy as np
import math
import matplotlib.pyplot as plt
import utils

dt = 0.01
num_rep = 1000

def cal_first_pass_time(m,d):
    qd = math.pow(math.pi, float(d) / 2) / math.gamma(float(d) / 2 + 1)
    statmat = np.zeros(shape=(m, d))
    t = 0
    while True:
        for i in range(m):
            statmat[i] += np.random.normal(0, np.sqrt(dt), d)
        statall = np.mean(statmat, 0)
        statmatscale = np.matrix(statmat - statall)
        Gammamat = sum([statmatscale[i].transpose() * statmatscale[i] for i in range(m)])
        t += dt
        # print(Gammamat/((m-1)*t*t))
        if np.linalg.det(Gammamat / ((m - 1) * t * t)) < 0:
            print(np.linalg.det(Gammamat / ((m - 1) * t * t)))
        vol = math.sqrt(np.linalg.det(Gammamat / ((m - 1) * t * t))) * qd
        if vol < 1:
            break
    return Gammamat, t

def cov_mat_cal(m,d):
    Gammamat, t  = cal_first_pass_time(m,d)
    asymp_sample = Gammamat/t
    return asymp_sample

def quantile_cal(m, d, alpha, error_quantile_scale = 1.2):
    samples =[]
    for i in range(num_rep):
        if i %100 ==0 and len(samples)!=0:
            tmp_quantile = np.percentile(samples, alpha *100)
            #print("current temp quantile estimation is {} aftger {} iterations".format(tmp_quantile, i))
        normal =  np.matrix(np.random.normal(0,np.sqrt(1./m), d))
        cov_mat = cov_mat_cal(m,d)
        sample = (m-1) * normal * np.linalg.inv(cov_mat) * normal.transpose()
        #print (sample)
        samples.append(sample)
    quantile = np.percentile(samples, alpha *100)
    error_quantile = quantile * error_quantile_scale
    error_prob = float(sum(i< error_quantile for i in samples))/num_rep - alpha
    return quantile, error_prob

def main():
    ms = [25,30, 35, 40, 45, 50]
    d = 2
    print("current d is {}".format(d))
    alphas = [0.95]
    output_means = []
    for m in ms:
        for alpha in alphas:
            num_sec = 2
            outputs = []
            for i in range(num_sec):
                output, error_prob = quantile_cal(m,d, alpha)
                outputs.append(output)
            CI_half_width  = np.std(outputs)/np.sqrt(num_sec)*2.262
            output_mean = np.mean(output)
            print("current m is {}, current alpha is {} ".format(m, alpha))
            print("quantile is {}, error probability as a function of m = {} is {}".format(output_mean, m, error_prob))
            u = utils.m_times_F_quantiles(m, d, alpha=1-alpha)
            print("corresponding F_quantile is {}".format(u))
            print("CI_half_width is {}".format(CI_half_width))
        #output_means.append(output_mean)
        #print(output_means)
    #plt.plot(ms, output_means, 'ro')
    #plt.xlabel('m')
    #plt.ylabel('quantile')
    #plt.show()
#ms = [5,6,7,8,9,10,15,20,25,30,35,40,45,50]
#  output_means = [26.12,6.42, 4.8,2.6,1.27,1.1404301413554883, 0.6033698639303676, 0.4096010325511576, 0.28959691516147484, 0.238784194308391, 0.1926816897623506, 0.16096472334474407, 0.15826298019804233, 0.13126220268878136]
if __name__ == "__main__":
    main()





