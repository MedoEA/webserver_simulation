import numpy as np

num_actions = 60
n_env = 1

mus_group1 = np.random.normal(loc=50, scale=2, size=(n_env, num_actions // 3)).flatten()
mus_group2 = np.random.normal(loc=60, scale=2, size=(n_env, num_actions // 3)).flatten()
mus_group3 = np.random.normal(loc=70, scale=2, size=(n_env, num_actions - (2 * num_actions // 3))).flatten()
mus = np.hstack((mus_group1, mus_group2, mus_group3))

sigmas_group1 = np.random.uniform(low=8, high=10, size=(n_env, num_actions // 3)).flatten()
sigmas_group2 = np.random.uniform(low=10, high=12, size=(n_env, num_actions // 3)).flatten()
sigmas_group3 = np.random.uniform(low=12, high=14, size=(n_env, num_actions - (2 * num_actions // 3))).flatten()
sigmas = np.hstack((sigmas_group1, sigmas_group2, sigmas_group3))

# one action to be dominant
dominant_action = np.random.randint(num_actions)
mus[dominant_action] = 40
sigmas[dominant_action] = 12