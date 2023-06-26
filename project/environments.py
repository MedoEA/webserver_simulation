import numpy as np

class Env:
    def __init__(self, mus, sigmas):
        self.mus = mus
        self.sigmas = sigmas
        self.env_key = 0
        
    def sample(self, action):
        mu, sigma = self.mus[action], self.sigmas[action]
        return np.random.normal(mu, sigma)
    
    def step(self, **kwargs):
        pass
    
    def best_action(self):
        return np.argmin(self.mus)
    
class DynamicEnv:
    def __init__(self, mus_all, sigmas_all, change_every):
        self.mus_all = mus_all
        self.sigmas_all = sigmas_all
        self.change_every = change_every
        self.timestep = 0
        self.reselect_env()
        
    def sample(self, action):
        mu, sigma = self.mus[action], self.sigmas[action]
        return np.random.normal(mu, sigma)
    
    def reselect_env(self):
        new_idx = np.random.randint(0, len(self.mus_all))
        self.env_key = new_idx
        self.mus, self.sigmas = self.mus_all[new_idx], self.sigmas_all[new_idx]
        
    def best_action(self):
        return np.argmin(self.mus)
        
    def step(self, **kwargs):
        self.timestep+=1 
        
        if self.timestep%self.change_every==0:
            self.reselect_env()