from utils import max_
import numpy as np
class ThompsonSamplingUnknownMeanVariance:

    def __init__(self, actions=20, mu_0=0, lambda_0=1, alpha_0=1, beta_0=1):
        self.actions = actions

        # Hyperparameters for the Normal-Gamma prior
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.counts = np.zeros(self.actions)
        self.sum_x = np.zeros(self.actions)
        self.sum_x_square = np.zeros(self.actions)

    def select_action(self):
        samples = np.zeros(self.actions)
        for k in range(self.actions):
            tau_k = np.random.gamma(self.alpha_0 + self.counts[k] / 2, 1 / (self.beta_0 + 0.5 * (self.sum_x_square[k] - (self.sum_x[k] ** 2) / (self.counts[k] + 1))))
            mu_k = np.random.normal(self.sum_x[k] / (self.lambda_0 + self.counts[k]), 1 / np.sqrt(tau_k * (self.lambda_0 + self.counts[k])))
            samples[k] = mu_k
        return self.max_(samples)

    def update(self, chosen_act, reward):
        self.counts[chosen_act] += 1
        self.sum_x[chosen_act] += reward
        self.sum_x_square[chosen_act] += reward ** 2