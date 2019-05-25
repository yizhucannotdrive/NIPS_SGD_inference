import sec_research_py_3 as sr
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import normaltest
hyper_par = np.array([2.0])
d = len(hyper_par)
theta0 = np.array([ 2.1])
r = 0.95
a = 1
m = 10
A = sr.A
update_mode = "iid_gaussian"

def alpha_0_cal(t):
    s = 1
    for i in range(1,t-1):
        tmp = 1
        for k in range(1,i):
            tmp *= (1- 2 * sr.stepsize_cal(k, r, a))
        s += tmp
    return s

def first_error_term_cal(t):
    delta_0 = np.array(hyper_par) - np.array(theta0)
    error =  np.linalg.norm( delta_0) * alpha_0_cal(t) / (a * np.sqrt(t))
    return error

## contour plot of 2 dim scaled statistisc
## qq plot of normality check
## shapiro test
## We can further consider independence and identity distributed for B(t_2)- B(t_1) vs B(t_3)- B(t_2)
def main():
    for oo in range(1000):
        print("current oo is {}".format(oo))
        p_val_normal =[]
        p_val_clt = []
        p_val_poly =[]
        num_iters = [100]
        Delta_t = 10
        num_rep = 100
        for num_iter in num_iters:
            samples = []
            samples_dim_list = []
            benchmarks = []
            DeltaB_1_list = []
            DeltaB_2_list = []
            diff_list = []
            for i in range(d):
                samples_dim_list.append([])
                DeltaB_1_list.append([])
                DeltaB_2_list.append([])
                diff_list.append([])

            for i in range(num_rep):
                current_theta = np.copy(theta0)
                current_theta_sum = np.copy(theta0)
                current_noise_sum = np.zeros(d)
                if i %100 == 0 and i!=0:
                    print(i)
                current_theta, current_theta_sum, current_noise_sum, T = sr.update_multi(current_theta, current_theta_sum, 0,
                                                                      num_iter - Delta_t,hyper_par, r, a, update_mode, current_noise_sum )
                pre_sample_sum = (current_theta_sum / (T + 1) - hyper_par) *(T+1)

                current_theta, current_theta_sum, current_noise_sum, T = sr.update_multi(current_theta, current_theta_sum, num_iter - Delta_t,
                                                                      Delta_t, hyper_par, r, a, update_mode, current_noise_sum)
                sample = (current_theta_sum/ (T+1) - hyper_par) * np.sqrt(T)
                noise_sample = current_noise_sum / (2*np.sqrt(T))
                diff = sample - 1./A * noise_sample
                samples.append(sample)
                sample_sum = (current_theta_sum / (T + 1) - hyper_par) * (T + 1)

                current_theta, current_theta_sum, current_noise_sum, T = sr.update_multi(current_theta, current_theta_sum, num_iter,
                                                        Delta_t, hyper_par, r, a, update_mode, current_noise_sum)
                post_sample_sum = (current_theta_sum / (T + 1) - hyper_par) * (T+1)

                for j in range(d):
                    samples_dim_list[j].append(sample[j])
                    DeltaB_1_list[j].append(sample_sum[j] - pre_sample_sum[j])
                    DeltaB_2_list[j].append(post_sample_sum[j] - sample_sum[j])
                    diff_list[j].append(diff[j])

                benchmarks.append((np.mean(np.random.random(T))-0.5) * np.sqrt(T))
            #strong approximation error check
            first_error_term = first_error_term_cal(num_iter)
            print("first term error is {} ".format(first_error_term))
            print("strong approximation second term error check:")
            for  j in range(d):
                print("dim {} error mean val is {}, error max val is {}, error min val is {}, error 95% quantile is {}, error 5% quantile is {}".
                      format(j, np.mean(diff_list[j]),np.max(diff_list[j]),np.min(diff_list[j]),  np.percentile(diff_list[j], 95), np.percentile(diff_list[j], 5)))
                per_more_1 = np.float(sum(i>1 for i in np.array(diff_list[j])))/len(diff_list[j])
                per_less_minus1 = np.float(sum(i<-1 for i in np.array(diff_list[j])))/len(diff_list[j])
                print("percentage more than 1 and less than -1 are {} and {}".format(per_more_1, per_less_minus1))


            """
            for j in range(d):
                for k in range(d):
                    cov = np.cov(np.array([DeltaB_1_list[j], DeltaB_2_list[k]]))[0][1]
                    print("covariance of dim {} vs dim {} is {}".format(j, k, cov))
            """
            # independence check, problematic??

            # normality check
            # for 2 dimensional case
            # Basic 2D density plot
            if d ==2:
                #exit()
                sns.set_style("white")
                sns.kdeplot(samples_dim_list[0], samples_dim_list[1], cmap="Blues", shade=True, shade_lowest=True)
                plt.show()
                sns.kdeplot(samples_dim_list[0], samples_dim_list[1])
                #plt.show()
            #samples_dim_list[0] = np.random.normal(1,2,len(samples_dim_list[0]))
            #samples_dim_list[0] = benchmarks
            for j in range(d):
                print("______________{}th normailty check_________________".format(j))
                tmp_mean = np.mean(samples_dim_list[j])
                tmp_std = np.std(samples_dim_list[j])
                print("sample mean and std are {} and {}".format(tmp_mean, tmp_std))
                plt.hist(np.array(samples_dim_list[j]), color='blue', edgecolor='black',
                         bins=int(10))
                sns.distplot(np.array(samples_dim_list[j]), hist=True, kde=False,
                             bins=int(10), color='blue',
                             hist_kws={'edgecolor': 'black'})
                #plt.show()
                sm.qqplot(np.array(samples_dim_list[j]), fit = True, line='45')
                #plt.show()
                print("Shapiro-Wilk Test:")
                stat, p = shapiro(np.array(samples_dim_list[j]))
                print('Statistics=%.3f, p=%.3f' % (stat, p))
                #compare with normal and CLT

                # interpret
                alpha = 0.05
                if p > alpha:
                    print('Sample looks Gaussian (fail to reject H0)')
                else:
                    print('Sample does not look Gaussian (reject H0)')
                normal_stat = []
                clt_stats = []
                for i in range(num_rep):
                    normal_samples = np.random.normal(0,1,T-Delta_t)
                    normal_stat.append(np.mean(normal_samples)* np.sqrt(T-Delta_t))
                    clt_samples = np.random.random(T-Delta_t)
                    clt_stats.append((np.mean(clt_samples)-0.5)* np.sqrt(T-Delta_t))
                stat_n, p_n = shapiro(normal_stat)
                stat_clt, p_clt = shapiro(clt_stats)
                print("num_iter is {}".format(num_iter))
                print('polyak_Statistics=%.3f, p=%.3f' % (stat, p))
                print('normal_Statistics=%.3f, p=%.3f' % (stat_n, p_n))
                print('cLt_Statistics=%.3f, p=%.3f' % (stat_clt, p_clt))
                p_val_normal.append(p_n)
                p_val_poly.append(p)
                p_val_clt.append(p_clt)
                # interpret
                alpha = 0.05
                print("D'Agostino K2 Test")
                stat, p = normaltest(samples_dim_list[j])
                if p > alpha:
                    print('Sample looks Gaussian (fail to reject H0)')
                else:
                    print('Sample does not look Gaussian (reject H0)')
                #plt.show()
    p_val_poly_mean = np.mean(p_val_poly)
    p_val_clt_mean = np.mean(p_val_clt)
    p_val_normal_mean = np.mean(p_val_normal)
    print("num of iter is {}".format(num_iter))
    print("p_Val_poly_mean, p_val_clt_mean, p_val_normal_mean are {}, {}, {}".format(p_val_poly_mean, p_val_clt_mean, p_val_normal_mean))




if __name__ == "__main__":
    main()
