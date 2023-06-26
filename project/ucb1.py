from utils import max_
import math


class UCB1():
    
    def __init__(self, c=1, actions=10):
        
        self.c = c
        self.actions = actions
        
        self.counts = [0 for col in range(self.actions)]
        
        self.values = [0.0 for col in range(self.actions)]
        
        self.action_total_reward = [0.0 for _ in range(self.actions)]
        self.action_avg_reward = [[] for action in range(self.actions)]
        
        return
    
    


    def select_action(self):
        actions = len(self.counts)
        for action in range(actions):
            if self.counts[action] == 0:
                return action
    
        ucb_values = [0.0 for action in range(actions)]
        total_counts = sum(self.counts)
        for action in range(actions):
            bonus =  self.c * (math.sqrt((2 * math.log(total_counts)) / float(self.counts[action])))
            ucb_values[action] = self.values[action] + bonus
        return max_(ucb_values)

    def update(self, chosen_act, reward):
        self.counts[chosen_act] = self.counts[chosen_act] + 1
        n = self.counts[chosen_act]
        
#     # Update average/mean value/reward for chosen action
        value = self.values[chosen_act]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        #new_value2 = value + (1/n * (reward - value))
        self.values[chosen_act] = new_value
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        return