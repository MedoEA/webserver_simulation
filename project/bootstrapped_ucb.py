from utils import max_
import numpy as np
class BootstrappedUCB:

    def __init__(self, B=200, delta=0.1, actions = 10):
        self.B = B
        self.delta = delta
        self.actions = actions

        self.counts = np.ones(self.actions)
        self.sum_rewards = np.random.normal(0, 1, self.actions)

    def select_action(self):
        means = self.sum_rewards / self.counts
        bootstrapped_quantiles = np.zeros(self.actions)
        for k in range(self.actions):
            bootstrapped_samples = np.random.normal(means[k], 1, (int(self.counts[k]), self.B))
            bootstrapped_quantiles[k] = np.percentile(bootstrapped_samples, 100 * (1 - self.delta))

        confidence_level = 1 / (sum(self.counts) + 1)
        second_order_correction = np.sqrt(np.log(2 / confidence_level) / self.counts)
        ucb_indices = means + bootstrapped_quantiles + second_order_correction

        return self.max_(ucb_indices)

    def update(self, chosen_act, reward):
        self.counts[chosen_act] += 1
        self.sum_rewards[chosen_act] += reward