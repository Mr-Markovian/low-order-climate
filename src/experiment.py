import numpy as np
from jax import jit
from omegaconf import DictConfig
import hydra

def run_experiment(cfg: DictConfig):
    """
    Runs the experiment by integrating trajectories using the provided solver.
    Instructions are fully defined in the cfg file.
    """

    # Load or generate initial conditions
    if cfg.run_type == "trajectory":
        X_o = np.expand_dims(np.load(cfg.initial_conditions_path), axis=1)
    elif cfg.run_type == "reach-attractor":
        model_dim = cfg.model_dim
        epsilon = cfg.epsilon
        X_o = np.random.multivariate_normal(
            np.zeros(model_dim), epsilon * np.eye(model_dim), size=cfg.num_initials
        ).T
    else:
        raise ValueError(f"Unknown initial condition generation method: {cfg.ic_generation.method}")

    # Time settings
    T_start = cfg.T_start
    T_stop = cfg.T_stop
    dt = cfg.dt
    dt_solver = cfg.dt_solver
    iters_delta_t = int(dt / dt_solver)
    model_dim = X_o.shape[0]

    # Instantiate the solver using hydra
    solver= hydra.utils.call(cfg.solver)

    # Initialize solver
    solver = jit(solver)

    # Initialize trajectories
    Trajs = np.zeros((int((T_stop - T_start) / dt), model_dim, cfg.num_initials))

    @jit
    def integrate_forward(X_):
        X = X_
        for _ in range(iters_delta_t):
            X = solver(x_initial=X)
        return X

    # Run simulation
    X_now = X_o
    Trajs[0] = X_now
    for i in range(1, Trajs.shape[0]):
        X_next = integrate_forward(X_now)
        Trajs[i] = X_next
        X_now = X_next

    # Save results
    output_path = f"{cfg.data_path}/Multiple_trajectories_N={cfg.num_initials}_gap={dt}_ti={T_start}_tf={T_stop}_dt_{dt}_dt_solver={dt_solver}.npy"
    np.save(output_path, Trajs[:, :, 0])
    print(f"Results saved to {output_path}")
    print("Job done")