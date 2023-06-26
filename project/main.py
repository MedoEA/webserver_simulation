from ucb1 import UCB1
from bootstrapped_ucb import BootstrappedUCB
from thompson_sampling import ThompsonSamplingUnknownMeanVariance
from environments import Env, DynamicEnv
from simulation import do_run
from methods import remove_none, remove_oldest_point, keep_uniform_and_top_points_re_evaluate
from gp import select_points_GP
from metrics import metrics
from initial_and_pb2 import InitialiseSample,InitialisePB2

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
# from ray import tune
# from ray.tune.schedulers.pb2 import PB2
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest.bayesopt import BayesOptSearch
# from hyperopt import hp
# from ray.tune.suggest.hyperopt import HyperOptSearch
import numpy as np
import scipy.stats as stats

from workload import mus, sigmas

def test_method(mus_all, sigmas_all, method, start_size=10, init_method=None, change_every=1000, verbose=True, plot=False):
    
    n_steps = 4000
    
    if len(mus_all.shape) == 1:
        mus_all = np.array([mus_all])
        sigmas_all = np.array([sigmas_all])
        change_every = n_steps + 1
        print("mus_all.shape: ", mus_all.shape)
    
    if not init_method:
        #new_points = np.random.beta(10, 100, size=start_size)
        new_points = np.random.uniform(low=0.0, high=0.5, size=start_size)
        #new_points = np.random.uniform(low=5, high=10, size=start_size)

    else:
        new_points = init_method(size=start_size)
    
    optimisation_steps = 20

    ## change functions here
    add_points_function    = method[0]
    remove_points_function = method[1]
        
    point_history = []
    score_history = []
    best_chosen_history = []
    all_points = np.array([])
    all_points_scores = np.array([])
    
    start = time.time()
    
    if len(mus_all.shape)==1:
        num_actions = mus_all.shape[0]
    else:
        num_actions = mus_all.shape[1]
        
    for optimisation_step in tqdm(range(optimisation_steps)):
        new_point_scores = []
        best_chosen_step = []
        avg_step =[]
        for point in new_points:
            #print("method_point=", point)
            response_times = []
            best_chosen = []
            for trial in range(5):
                if len(mus_all.shape)==1:
                    env = Env(mus_all, sigmas_all)
                    #env = NoisyEnv(mus, sigmas, noise_std=0.05)
                    
                else:
                    env = DynamicEnv(mus_all, sigmas_all, change_every=change_every)
                
                #env = Env(mus, sigmas)
                select_action_class = UCB1(actions = num_actions, c=point)

                results = do_run(env, 
                                select_action_class,
                                n_steps=n_steps)
                response_times.append(results["average_response_time"])
                best_chosen.append(results["best_chosen"])
            new_point_scores.append(np.mean(response_times))
            avg_step.append(response_times)
            best_chosen_step.append(best_chosen)
            
        best_chosen_history.append(best_chosen_step)
        
        point_history.append(new_points)
        
        score_history.append(new_point_scores)

        all_points = np.concatenate((all_points, new_points))

        all_points_scores = np.concatenate((all_points_scores, new_point_scores))

        # add function
        new_points = add_points_function(all_points, all_points_scores)

        # remove points function
        if remove_points_function:
            all_points, all_points_scores = remove_points_function(all_points, all_points_scores)
        
    
    run_time = time.time() - start 
    avg_step= np.array(avg_step)
    if plot:
        plt.plot(avg_step.mean(axis=0).mean(axis=0))
    scores = {}
    if verbose:
        print(f"select_new_points: {method[0].__name__} - remove_points: {method[1].__name__}")
        print(f"run_time: {run_time}")
    scores["run_time"] =  run_time
    for metric_name, metric_func in metrics.items():
        score = metric_func(score_history)
        scores[metric_name] = score
        if verbose:
            print(f"{metric_name}: {score}")
        
    return scores, avg_step.mean(axis=0), best_chosen_step


def test_method_n_times(n,plot=False, **kwargs):
    results = []
    response_times_means = []
    best_chosen_step_prob = []
    for i in range(n):
        result, response_times_mean,best_chosen_step = test_method(**kwargs)
        results.append(result)
        response_times_means.append(response_times_mean)
        best_chosen_step_prob.append(best_chosen_step)
    mean_response_times = np.array(response_times_means)
    #best_action_plot = np.array(best_chosen_step_prob).mean(axis=0)
    best_action_plot = np.array(best_chosen_step_prob)
    if plot:
        plt.figure()
        plt.title("Mean response time")
        plt.plot(mean_response_times.mean(axis=0).mean(axis=0))
        plt.figure()
        plt.title("probability")
        plt.plot(best_action_plot.mean(axis=0).mean(axis=0).mean(axis=0))
    arrays = {key:[] for key in result.keys()}
    for item in results:
        for key, value in item.items():
            arrays[key].append(value)
    return {key:np.mean(item) for key, item in arrays.items()},mean_response_times,best_action_plot


def main():

    

    add_1_remove_none = (select_points_GP, remove_none)
    add_1_remove_oldest = (select_points_GP, remove_oldest_point)
    PB2_method = (select_points_GP, remove_none)
    add_1_keep_uni_dy = (select_points_GP,keep_uniform_and_top_points_re_evaluate)

    
    
    
    means_IL,mean_r_IL, best_c_IL = test_method_n_times(3, 
                    mus_all=mus, 
                    sigmas_all=sigmas, 
                    init_method=InitialiseSample,
                    method=add_1_remove_none, 
                    verbose=True,
                    plot=False)
    means_DF,mean_r_DF, best_c_DF = test_method_n_times(3, 
                    mus_all=mus, 
                    sigmas_all=sigmas, 
                    init_method=InitialiseSample,
                    method=add_1_remove_oldest, 
                    verbose=True,
                    plot=False)
    means_pillar,mean_r_pillar, best_c_pillar = test_method_n_times(3, 
                    mus_all=mus, 
                    sigmas_all=sigmas, 
                    init_method=InitialiseSample,
                    method=add_1_keep_uni_dy, 
                    verbose=True,
                    plot=False)
    # m4,response_times_mean_pb2, P_best_PB2 = test_method_n_times(3,
    #                                       mus_all=mus,
    #                                       sigmas_all=sigmas,
    #                                       init_method=InitialisePB2,
    #                                       method=PB2_method,
    #                                       verbose=True,
    #                                       plot=False)
    
    
    
    
    #########################################################
    # Statistical Analysis:
    mean_res_IL = [np.mean(item) for item in mean_r_IL]
    mean_res_pillar = [np.mean(item)  for item in mean_r_pillar]

    t_statistic, p_value = stats.ttest_ind(mean_res_IL,
                                        mean_res_pillar)

    print("T-statistic: ", t_statistic)
    print("P-value: ", p_value)

    alpha = 0.05

    if p_value < alpha:
        print("There is a significant difference between Update UCB and BO-UCB methods. ")
        print(f"The p_value is {p_value:.4f}")
        
    else:
        print("There is no significant difference between Update UCB and BO-UCB methods. ")
        print(f"The p_value is {p_value:.4f}, indicating a small difference between the methods.")

    mean_res_Df = [np.mean(item) for item in mean_r_DF]
    mean_res_pillar = [np.mean(item)  for item in mean_r_pillar]

    t_statistic, p_value = stats.ttest_ind(mean_res_Df,
                                        mean_res_pillar)

    print("T-statistic: ", t_statistic)
    print("P-value: ", p_value)

    ########

    if p_value < alpha:
        print("There is a significant difference between Discard UCB and BO-UCB methods. ")
        print(f"The p_value is {p_value:.4f}")
    else:
        print("There is no significant difference between Discard UCB and BO-UCB methods. ")
        print(f"The p_value is {p_value:.4f}, indicating a small difference between the methods.")

    mean_res_Df = [np.mean(item) for item in mean_r_DF]
    mean_res_IL = [np.mean(item) for item in mean_r_IL]

    t_statistic, p_value = stats.ttest_ind(mean_res_Df,
                                        mean_res_IL)

    print("T-statistic: ", t_statistic)
    print("P-value: ", p_value)

    ##########

    if p_value < alpha:
        print("There is a significant difference between Discard UCB and Update UCB methods. ")
        print(f"The p_value is {p_value:.4f}")
    else:
        print("There is no significant difference between Discard UCB and Update UCB methods. ")
        print(f"The p_value is {p_value:.4f}, indicating a small difference between the methods.")
    #######################################################
    # Average response time plot

    CI_I= 1.96 * np.std(mean_r_IL)/np.sqrt(3)
    update_ucb = mean_r_IL.mean(axis=0).mean(axis=0)
    CI_D= 1.96 * np.std(mean_r_DF)/np.sqrt(3)
    discard_ucb = mean_r_DF.mean(axis=0).mean(axis=0)
    CI_P= 1.96 * np.std(mean_r_pillar)/np.sqrt(3)
    BO_ucb = mean_r_pillar.mean(axis=0).mean(axis=0)
    # CI_PB2= 1.96 * np.std(response_times_mean_pb2)/np.sqrt(3)
    # PB2_m = response_times_mean_pb2.mean(axis=0).mean(axis=0)

    plt.figure()
    plt.plot(update_ucb,label='Update_UCB' ,lw=2,ls='--', color= 'g')
    plt.fill_between(np.arange(len(update_ucb)), (update_ucb - CI_I),(update_ucb + CI_I),color= 'g', alpha=0.1 )
    plt.plot(discard_ucb,label='Discard_UCB' ,lw=2,ls='--', color= 'r')
    plt.fill_between(np.arange(len(discard_ucb)), (discard_ucb - CI_D),(discard_ucb + CI_D),color= 'r', alpha=0.1 )
    plt.plot(BO_ucb,label='BO_UCB' ,lw=2,ls='--', color= 'purple')
    plt.fill_between(np.arange(len(BO_ucb)), (BO_ucb - CI_P),(BO_ucb + CI_P),color= 'purple', alpha=0.1 )
    # plt.plot(PB2_m,label='PB2' ,lw=2,ls='--', color= '#819999')
    # plt.fill_between(np.arange(len(PB2_m)), (PB2_m - CI_PB2),(PB2_m + CI_PB2),color= '#819999', alpha=0.1 )

    plt.xlabel("Time step")
    plt.ylabel("Average response time")
    #plt.title("UCB1 Simulation")
    #plt.xlim([-10,250])
    plt.legend()
    plt.show()

    # Best Action Plot
    
    mean_IL = best_c_IL.mean(axis=0).mean(axis=0).mean(axis=0)
    std_dev_IL = best_c_IL.std(axis=0).mean(axis=0).mean(axis=0)
    # Calculate the 95% confidence interval
    conf_interval_IL = 1.96 * std_dev_IL / np.sqrt(best_c_IL.shape[0])

    mean_DF = best_c_DF.mean(axis=0).mean(axis=0).mean(axis=0)
    std_dev_DF = best_c_DF.std(axis=0).mean(axis=0).mean(axis=0)
    # Calculate the 95% confidence interval
    conf_interval_DF = 1.96 * std_dev_DF / np.sqrt(best_c_DF.shape[0])

    mean_P = best_c_pillar.mean(axis=0).mean(axis=0).mean(axis=0)
    std_dev_P = best_c_pillar.std(axis=0).mean(axis=0).mean(axis=0)
    # Calculate the 95% confidence interval
    conf_interval_P = 1.96 * std_dev_P / np.sqrt(best_c_pillar.shape[0])

    # mean_PB2 = P_best_PB2.mean(axis=0).mean(axis=0).mean(axis=0)
    # std_dev_PB2 = P_best_PB2.std(axis=0).mean(axis=0).mean(axis=0)
    # # Calculate the 95% confidence interval
    # conf_interval_PB2 = 1.96 * std_dev_PB2 / np.sqrt(P_best_PB2.shape[0])

    plt.figure()
    plt.plot(mean_IL, label='Update_UCB' ,lw=2,ls='-', color= 'g')
    plt.fill_between(range(len(mean_IL)), (mean_IL-conf_interval_IL), (mean_IL+conf_interval_IL), color='g', alpha=.1)
    plt.plot(mean_DF, label='Discard_UCB' ,lw=2,ls='-', color= 'r')
    plt.fill_between(range(len(mean_DF)), (mean_DF-conf_interval_DF), (mean_DF+conf_interval_DF), color='r', alpha=.1)
    plt.plot(mean_P, label='BO_UCB' ,lw=2,ls='-', color= 'purple')
    plt.fill_between(range(len(mean_P)), (mean_P-conf_interval_P), (mean_P+conf_interval_P), color='purple', alpha=.1)
    # plt.plot(mean_PB2, label='PB2' ,lw=2,ls='-', color= '#819999')
    # plt.fill_between(range(len(mean_PB2)), (mean_PB2-conf_interval_PB2), (mean_PB2+conf_interval_PB2), color='#819999', alpha=.1)
    plt.xlabel("Time step")
    plt.ylabel("Probability of choosing the best action")
    plt.legend()
    plt.show()



    
    


if __name__ == "__main__":
        main()