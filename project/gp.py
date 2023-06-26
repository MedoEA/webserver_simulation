from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import numpy as np


def select_points_GP(point_set, point_scores):
    '''
    Accepts a set of points, and a set of scores, 
    and returns a new set of points
    '''
    model1 = GaussianProcessRegressor()

    # fit the model
    model1.fit(point_set.reshape(-1, 1), point_scores)

    new_points = np.random.beta(10, 100, size=10)

    original_points_predictions, _ = model1.predict(point_set.reshape(-1, 1), return_std=True)
    new_points_predictions, std   = model1.predict(new_points.reshape(-1, 1), return_std=True)

    best = min(original_points_predictions)
    mu = new_points_predictions

    z = (mu - best) / (std+1E-9)
    probs = (mu - best)* norm.cdf(z) + (std+1E-9) * norm.pdf(z)

    ind_best_point = np.argmin(probs)
    new_point = new_points[ind_best_point]

    return np.array([new_point])
