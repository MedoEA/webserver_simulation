import numpy as np
import ray
import argparse
from ray import air, tune
from ray.tune.schedulers.pb2 import PB2
#from ray.tune.suggest import ConcurrencyLimiter
#from ray.tune.suggest.bayesopt import BayesOptSearch
#from hyperopt import hp
#from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import run, sample_from
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.tune_config import TuneConfig
from ray.air.config import FailureConfig, RunConfig
from ray.tune.tuner import Tuner
import warnings
warnings.simplefilter('ignore')

def easy_objective_pb2(config):
    # Hyperparameters
    c = config["c"]

    for step in range(config["steps"]):
        # Iterative training function - can be an arbitrary training procedure
        #intermediate_score = resp_time(step, c)
        intermediate_score = c
        # Feed the score back back to Tune.
        session.report({"iterations": step, "mean_loss": intermediate_score})


def initialise_PB_point():
    perturbation_interval = 5
    pbt = PB2(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_bounds={
            # hyperparameter bounds.
            "c": [0.0001, 0.02],
        },
    )

    tuner = tune.Tuner(
        easy_objective_pb2,
        run_config=air.RunConfig(
            name="pbt_test",
            verbose=0,
            stop={
                "training_iteration": 15,
            },
            failure_config=air.FailureConfig(
                fail_fast=True,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            metric="mean_loss",
            mode="min",
            num_samples=6,
        ),
        param_space={
            "steps": 10,
            "c": sample_from(
                    lambda spec: np.random.uniform(0.1, 0.5)),
            "some_other_factor": 1,
            # This parameter is not perturbed and is used to determine
            # checkpoint frequency. We set checkpoints and perturbations
            # to happen at the same frequency.
            "checkpoint_interval": perturbation_interval,
        },
    )

    pb2_results = tuner.fit()

    points = list(pb2_results.get_best_result().config.values())[1]
    
    return [points]
    
def InitialiseSample(size):
    return np.random.uniform(low=0.0, high=0.5, size=size)

def InitialisePB2(size):
    points = []
    for i in range(size):
        points += initialise_PB_point()
    return points
