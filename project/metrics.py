import numpy as np

def average_response_time_end_all_points(score_history):
    return np.mean(score_history[-1])

def average_response_time_best_point_final_step(score_history):
    return min(score_history[-1])

def average_response_time_all_points_all_steps(score_history):
    return np.concatenate(score_history).mean()

metrics = {"average_response_time_end_all_points": average_response_time_end_all_points,
           "average_response_time_best_point_final_step": average_response_time_best_point_final_step,
           "average_response_time_all_points_all_steps": average_response_time_all_points_all_steps,
          }
