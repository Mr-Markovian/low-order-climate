"""A vectorized code to compute multiple trajectories at simultaneously. The input has the shape
[time,dimension,no. of intial conditions]"""
from functools import partial
from jax import config
import numpy as np
from ode_solvers import rk4_solver as solver
from testbed_models import L86_gcm
import hydra
from omegaconf import DictConfig
from experiment import run_experiment
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

config.update("jax_enable_x64", True)

@hydra.main(config_path="../config", config_name="base")
def main(cfg: DictConfig):
    # Instantiate and run the experiment using Hydra
    run_experiment(cfg.experiment)
    
if __name__ == "__main__":
    main()
