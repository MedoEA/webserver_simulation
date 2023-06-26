def do_run(env, 
           select_action_class,
           n_steps,
            ): 
    selected_actions, rewards, res, Nom_norm, cum_response_time, cum_cost, average_response_time, env_keys, best_chosen = [], [],[],[],[],[],[],[],[]
    for i in range(n_steps):
        
        # select the action using the given method
        action = select_action_class.select_action()
        
        best_action = env.best_action()
        best_action_chosen = action == best_action
        best_chosen.append(best_action_chosen)
        
        # this is the enviroment applying a delay sampled from using the mu and sigma 
        env_keys.append(env.env_key)
        delay = env.sample(action)
        reward = 1-delay
        
        # log the result of the selected actions
        selected_actions.append(action)
        rewards.append(reward)
        res.append(delay)
        
        average_response_time.append(sum(res)/len(res))
        cum_response_time.append(sum(res))
        
        # finally, we update our class
        select_action_class.update(action, reward)
        
        env.step()
        
        
        
    return {"rewards": rewards, 
            "cum_response_time": cum_response_time, 
            "average_response_time": average_response_time, 
            "res": res, 
            "best_chosen" : best_chosen,
            "selected_actions": selected_actions,
            "env_keys": env_keys}
            