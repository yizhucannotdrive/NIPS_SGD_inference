import numpy as np
import math
import utils
import cPickle as pickle
import os

us = dict()
us[1] = dict()
us[2] = dict()
us[10] = dict()
us[1][10] = 0.693
us[1][20] = 0.251
us[1][30] = 0.151
us[2][10] = 1.265
us[2][20] = 0.399
us[2][30] = 0.24
us[10][25] = 1.678
us[10][30] = 1.175
us[10][35] = 0.887
us[10][15] = 13.93
us[10][20] = 3.078
A = 2
#sigma = 0.7
sigma = 1
print("x  sigma  is {}".format(sigma))
def stepsize_cal(t, r, a):
    stepsize = a/np.power(t, r)
    #if stepsize < 0.0001:
    #    stepsize *= 10
    #stepsize = 0.1
    return stepsize

## assume X is a gaussian with mean 0 and identity covariance matrix, Y= \beta * X + \epsilon, \epsilon is a normal(0,1)

# update mode here is "sgd"
def one_update_time_t(current_theta,t, r,a, hyper_par, const = A, model = "linear"):
    #calculate stepsize as a*t^(-r)
    stepsize = stepsize_cal(t, r, a)
    #stepsize = 0.01
    #update equation depends on the parameter of this gaussian
    if model == "linear":
        xx = np.random.normal(0,1,len(current_theta))
        yy = np.dot(xx, hyper_par) + np.random.normal(0,0.1)
        #print(current_theta, xx, yy, const, stepsize, model)
        next_theta = current_theta - stepsize * const * np.dot((np.dot(xx,current_theta) - yy), xx)
        noise = const * np.dot((np.dot(xx,current_theta) - yy), xx) - const * (current_theta - hyper_par)
        #print(next_theta)
        return next_theta, noise
    # X is from a normal, Y is bernouli taking values[-1,1]
    if model == "logistic":
        #xx = np.random.normal(1,1,len(current_theta))
        xx = 2*np.random.random(len(current_theta))-1
        xx = np.random.normal(0,sigma, len(current_theta))
        #xx = np.ones(len(current_theta))
        p_y_1 = 1./ (1+np.exp(-np.dot(xx, hyper_par)))
        yy = np.random.binomial(1, p_y_1)*2-1
        #print(xx, p_y_1, yy)
        next_theta = current_theta + stepsize * np.dot(1./ (1+np.exp( yy * np.dot(xx, current_theta))) * yy, xx)
        return next_theta, None



# update mode here is "iid_gaussian"
def one_update_time_t_iid_noise(current_theta,t, r, a, hyper_par, const = A, model = "linear"):
    if model!= "linear":
        raise("iid gaussian noise case, mode has to be linear regression")
    #calculate stepsize as a*t^(-r)
    stepsize = stepsize_cal(t, r, a)
    #stepsize = 0.01
    noise = np.random.normal(0,1,len(current_theta))
    next_theta = current_theta - stepsize * (const * (current_theta - hyper_par) + noise)
    #print(next_theta)
    return next_theta, noise


def update_multi(current_theta, current_theta_sum, current_index, update_times, hyper_par, r, a, update_mode = "sgd", current_noise_sum =0, model = "linear"):
    for i in range(current_index, current_index + update_times):
        if update_mode == "sgd":
            current_theta, current_noise = one_update_time_t(current_theta, i+1, r, a, hyper_par,model= model)
        elif update_mode == "iid_gaussian":
            current_theta, current_noise = one_update_time_t_iid_noise(current_theta, i + 1, r, a, hyper_par, model = model)
        else:
            current_noise = np.random.normal(0,1,len(current_theta))
            #current_noise = np.random.random(len(current_theta))
            current_theta = current_noise + hyper_par
        if model == "linear":
            current_theta_sum += current_theta
            current_noise_sum += current_noise
        else:
            current_theta_sum += current_theta
        #if i%1000==0:
        #    print(current_theta)
        #print(current_theta)
        #if i%10000==0:
        #    print("current theta is {}, current pol_ave is {}".format(current_theta, current_theta_sum/(i+1)))
    return current_theta, current_theta_sum, current_noise_sum, current_index + update_times

# assume delay function equals  t^(-0.6*d)

def atfunction(t,d):
    eta = (0.5+0.1)*d
    return np.power(t,-eta)
    #return 0



def Seq_CI(d, hyper_par, theta0, m, u, epsilon, update_mode, r, a, budget, Delta, samplesize, num_rep):
    truevalue = hyper_par
    qd = math.pow(math.pi,float(d)/2)/math.gamma(float(d)/2+1)
    sum_in_cr = 0
    sum_in_cr_noise = 0
    num_reach_budget = 0
    T_record = []
    ratios = []
    num_ratios = []
    denom_ratios = []
    ratio_more_than_1_num = 0
    for j in range(num_rep):
        if j%(num_rep/10) == 0:
            print(j)
        sim_out = []
        thetas = []
        noises_s =[]
        for i in range(m):
            theta = np.copy(theta0)
            Thetassum = np.copy(theta0)
            theta, Thetasum, noise_sum, T = update_multi(theta, Thetassum, 0, samplesize, hyper_par, r, a, update_mode = update_mode)
            #print(Thetassum)
            sim_out.append(Thetassum/(T+1))
            thetas.append(theta)
            noises_s.append(noise_sum/(T+1))
        T=T+Delta
        ind=1
        #print("after initialization stage, polyak avearge estimations are {}".format(sim_out))
        while ind==1:
            if T%budget ==0:
                print("current step is {}".format(T))
            for i in range(m):
                #sim_out[i]=np.concatenate((sim_out[i],np.random.exponential(1,samplesize)))
                theta = np.copy(thetas[i])
                Thetassum = np.copy(sim_out[i]*(T-Delta+1))
                noise_sum = np.copy(noises_s[i]*(T-Delta+1))
                theta, Thetasum, noise_sum,T = update_multi(theta, Thetassum, T-Delta, Delta,  hyper_par, r, a, update_mode = update_mode, current_noise_sum=noise_sum)
                sim_out[i] = np.copy(Thetassum/(T+1))
                noises_s[i] = np.copy(noise_sum/(T+1))
                thetas[i] = theta
            statmat=np.zeros(shape=(m,d))
            noisemat =  np.zeros(shape=(m,d))
            for i in range(m):
                statmat[i]=sim_out[i]
                noisemat[i] = 1./A * noises_s[i]
            statall=np.mean(sim_out,0)
            noise_mean =  np.mean(noises_s,0)
            #print("True",statall - truevalue)
            #print("noise",noise_mean)
            statmatscale = np.matrix(statmat-statall)
            noisescale = np.matrix(noisemat-noise_mean)
            Gammamat = sum([ statmatscale[i].transpose() *statmatscale[i] for i in range(m)])/(m-1)
            noise_Gammamat =  sum([ noisescale[i].transpose() *noisescale[i] for i in range(m)])/(m-1)
            #if T>100000:
                #print(math.sqrt(np.linalg.det(Gammamat))*qd*np.power(u,d/2)+atfunction(T,d))
            #print(math.sqrt(np.linalg.det(Gammamat))*qd*np.power(u,d/2)+atfunction(T,d)<epsilon)
            #print(math.sqrt(np.linalg.det(Gammamat))*qd*np.power(u,d/2)+atfunction(T,d))
            vol = (math.sqrt(np.linalg.det(Gammamat)) * qd + atfunction(T,d)) * np.power(u,float(d)/2)
            vol_noise = (math.sqrt(np.linalg.det(noise_Gammamat)) * qd + atfunction(T,d)) * np.power(u,float(d)/2)
            if  vol < epsilon:
                ind=0
            else:
                T=T+Delta
            if T>budget:
                #print("exceed budget")
                #print(math.sqrt(np.linalg.det(Gammamat))*qd*np.power(u,d/2)+atfunction(T,d))
                ind=0
                num_reach_budget+=1
        # line 132-133  only works when d=1
        if d==1:
            numer_ratio = np.power(np.array((statall - truevalue))/(1./A *np.mean(noises_s, 0)),2)
            denumer_ratio = Gammamat/noise_Gammamat
            ratio = numer_ratio/ denumer_ratio
            if ratio > 1:
                ratio_more_than_1_num += 1
        #print("numer_ratio, denumer_ratio, ratio are {}, {}, {}".format(numer_ratio, denumer_ratio, ratio))
            num_ratios.append(numer_ratio)
            denom_ratios.append(denumer_ratio)
            ratios.append(ratio)
        plug_in_stat = np.matrix(statall-truevalue)*np.linalg.inv(Gammamat)*np.matrix(statall-truevalue).transpose()
        plug_in_noise_stat = np.matrix(1./A *np.mean(noises_s, 0))*np.linalg.inv(noise_Gammamat)*np.matrix(1./A *np.mean(noises_s, 0)).transpose()
        if plug_in_stat < u:
            sum_in_cr += 1
        if plug_in_noise_stat<u:
            sum_in_cr_noise += 1
        #else:
        #    print("CI true value plug in statistic vs u are {} and {}".format(plug_in_stat, u))
        if T< budget:
            T_record.append(T)

    #print("estimate from each sections are {}".format(sim_out))

    #print("epsilon equals {}, num of sections equals {}".format(epsilon, m ))
    cov_rate = float(sum_in_cr)/num_rep
    #print("coverage rate is {}". format(cov_rate))

    if d==1:
        ratio_mean = np.mean(ratios)
        numer_ratio_mean = np.mean(num_ratios)
        denom_ratio_mean = np.mean(denom_ratios)
        ratio_std = np.std(ratios)
        numer_ratio_std = np.std(num_ratios)
        denom_ratio_std = np.std(denom_ratios)
    #print("ratio mean, num_ratio_mean, denum_ratio_mean are {}, {}, {}".format(ratio_mean, numer_ratio_mean, denom_ratio_mean))
    #print("ratio std, num_ratio_std, denum_ratio_std are {}, {}, {}".format(ratio_std, numer_ratio_std, denom_ratio_std))
    #print("real/designed ratio greater than 1 number is {}".format(ratio_more_than_1_num))
    cov_rate_noise = float(sum_in_cr_noise)/num_rep
    print("coverage using what is supposed to be used is {}".format(cov_rate_noise))

    if len(T_record)!=0:
        ave_T_record = np.mean(T_record)
        #print("average sim time is {}".format(ave_T_record))
        scaled_ave_T_record = np.power(epsilon,2./d) * ave_T_record
        #print("scaled average sim time is {}".format(scaled_ave_T_record))
    #print("num of experiment reaches computation budget is {} out of {}".format(num_reach_budget, num_rep))
        return cov_rate, ave_T_record, scaled_ave_T_record
    else:
        raise("length of T_record is 0")

def main():
    r = 0.95
    a = 1
    budget = 500000
    Delta = 10
    samplesize = 10
    num_rep = 1000
    hyper_pars = dict()
    hyper_pars [1] = [1.0]
    hyper_pars [2] = [1.0, 2.0]
    theta0s = dict()
    theta0s[1] = [0.9]
    theta0s[2] = [0.9, 1.9]
    ds = [1,2]
    ms = [10,20,30]
    epsilon_ori_s = [0.5, 0.1, 0.05, 0.01]
    update_modes = ["iid_gaussian", "sgd"]

    hyper_pars = dict()
    hyper_pars[10] = [1.0, 1.1, 1.2, 1.5, 1.4, 1.6, 1.7, 1.9, 1.8 ,1.3]
    theta0s = dict()
    theta0s [10] = [0.9, 1.15, 1.12, 1.57, 1.21, 1.34, 1.56, 1.45, 1.89 ,1.13]
    ds =[10]
    ms = [20,25,30,35]
    epsilon_ori_s = [0.5,0.1,0.05, 0.01]
    #update_modes = ["iid_gaussian"]
    #update_modes = ["else"]




    #epsilon_ori_s = [0.5, 0.1]
    # update_mode = "benchmark"
    result = {}
    cov_F_op = True
    cov_op = False
    for d in ds:
        hyper_par = hyper_pars[d]
        theta0 =  theta0s[d]
        for m in ms:
            for epsilon_ori in epsilon_ori_s:
                epsilon = np.power(epsilon_ori, d)
                for update_mode  in update_modes:
                    key = "m is {}, d is {}, update mode is {}, epsilon is {}".format(m, d, update_mode, epsilon)
                    result[key] = {}
                    if cov_op:
                        u = us[d][m]
                        #test stopping time of k(\epsilon) ans k_R(\epsilon)
                        #u = 1
                        #brutal force searched scaling parameter for pre-limit
                        #u=0.51
                        coverage, sim_sum, sim_scale_sum = Seq_CI(d, hyper_par, theta0, m, u, epsilon, update_mode, r, a, budget, Delta, samplesize, num_rep)
                        result[key] ["coverage"] = coverage
                        result[key]["sim_sum"] = sim_sum
                        result[key]["sim_scale_sum"] = sim_scale_sum
                        print("***********************************************************************")
                        print(key)
                        print("coverage is {}".format(result[key]["coverage"]))
                        print("sim_sum is {}".format(result[key]["sim_sum"]))
                        print("sim_scale_sum is {}".format(result[key]["sim_scale_sum"]))
                        print("***********************************************************************")
                    if cov_F_op:
                        u = utils.m_times_F_quantiles(m, d, alpha=0.05)
                        coverage_F, _, _ = Seq_CI(d, hyper_par, theta0, m, u, epsilon, update_mode, r, a,
                                                              budget, Delta, samplesize, num_rep)
                        result[key]["coverage_F"] = coverage_F
                        print("***********************************************************************")
                        print(key)
                        print("coverage_F is {}".format(result[key] ["coverage_F"]))
                        print("***********************************************************************")
    exit()
    fname = os.path.join(".", "result.p")
    pickle.dump(result, open(fname, "wb"))





if __name__ == "__main__":
    main()