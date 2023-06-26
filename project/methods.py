import numpy as np
from workload import mus, sigmas


def remove_none(point_set, all_points_scores):
    return point_set, all_points_scores

def remove_oldest_point(point_set, all_points_scores):
    return point_set[1:], all_points_scores[1:]



def keep_uniform_and_top_points_re_evaluate(point_set, all_points_scores,mus_all=mus,sigmas_all=sigmas, initial_points=None, uniform_fraction=0.5, re_evaluation_period=2):
    
    from environments import Env, DynamicEnv
    from ucb1 import UCB1
    from simulation import do_run
    

    if len(mus_all.shape)==1:
        num_actions = mus_all.shape[0]
    else:
        num_actions = mus_all.shape[1]

    if initial_points is None:
        initial_points = np.random.uniform(low=5, high=10, size=20)
    threshold_index = int(len(initial_points) * 0.8)

    if threshold_index == 0:
        return point_set, all_points_scores

    # Calculate the uniform spacing
    spacing = max(1, len(point_set) // threshold_index)

    # Choose points with uniform spacing and top-performing points
    n_uniform_points = int(threshold_index * uniform_fraction)
    n_top_points = threshold_index - n_uniform_points

    uniform_indices = np.arange(0, len(point_set), spacing)[:n_uniform_points]
    top_indices = np.argsort(all_points_scores)[:n_top_points]
    selected_indices = np.union1d(uniform_indices, top_indices)

    selected_point_set = point_set[selected_indices]

    # Re-evaluate the selected points and update their average response time only if the current iteration is a multiple of re_evaluation_period
    if len(point_set) % re_evaluation_period == 0:
        new_selected_scores = []
        for point in selected_point_set:
            response_times = []
            for trial in range(1):
                env = Env(mus_all, sigmas_all)
                select_action_class = UCB1(actions=num_actions, c=point)

                results = do_run(env, select_action_class, n_steps=2000)
                response_times.append(results["average_response_time"])

            new_selected_scores.append(np.mean(response_times))

        # Update the all_points_scores with the new selected scores
        all_points_scores[selected_indices] = new_selected_scores

    # Remove the bottom points
    point_set = np.delete(point_set, np.setdiff1d(np.arange(len(point_set)), selected_indices))
    all_points_scores = np.delete(all_points_scores, np.setdiff1d(np.arange(len(all_points_scores)), selected_indices))

    return point_set, all_points_scores
