import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
# set seed
np.random.seed(42)

from kinematics.planar_arms import PlanarArms

from network.reservoir import RCNetwork
from utils import cumulative_sum, safe_save
from contextlib import contextmanager


# supress standard output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def trial(ArmModel: PlanarArms,
          ReservoirModel: RCNetwork,
          training: bool,
          scale_out: float,
          do_reset: bool = True,
          arm: str | None = None,
          min_movement_time: int = 60,
          max_movement_time: int = 80,
          t_wait: int | None = 20,
          learn_delta: int = 5,
          noise: float = 0.0):

    if t_wait is None:
        t_wait = learn_delta + 1

    if arm is None:
        arm = np.random.choice(['left', 'right'])

    ArmModel.move_randomly(arm=arm,
                           t_min=min_movement_time,
                           t_max=max_movement_time,
                           t_wait=t_wait)

    if arm == 'right':
        thetas = np.array(ArmModel.trajectory_thetas_right)
        end_effectors = np.array(ArmModel.end_effector_right)
    else:
        thetas = np.array(ArmModel.trajectory_thetas_left)
        end_effectors = np.array(ArmModel.end_effector_left)

    if noise > 0.0:
        thetas += np.random.normal(0, noise, size=thetas.shape)

    input_gradients = ArmModel.calc_gradients(array=thetas, delta_t=learn_delta)
    targets = ArmModel.calc_gradients(array=end_effectors, delta_t=learn_delta) * scale_out

    # train reservoir based on input and target
    prediction = ReservoirModel.run(data_in=input_gradients,
                                    data_target=targets,
                                    rls_training=training,
                                    do_reset=do_reset)

    # reset trajectories
    ArmModel.clear()

    return prediction, targets


def run_force_training(simID: int,
                       N_trials_training: int,
                       N_trials_test: int,
                       scale_in: float = 10.0,
                       scale_out: float = 0.01,
                       reservoir_g: float = 1.2,
                       reservoir_alpha: float = 0.2,
                       reservoir_dim: int = 500,
                       reservoir_rec_prop: float = 0.2,
                       noise: float = 0.0,
                       reset_after_epoch: bool = True,
                       moving_arm: str | None = 'right',
                       do_plot: bool = False,
                       fb_con: bool = True):

    # save results in...
    results_folder = f'results/run_{simID}/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # create arms
    arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)

    reservoir = RCNetwork(dim_reservoir=reservoir_dim,
                          dim_in=2,
                          dim_out=2,
                          sigma_rec=reservoir_rec_prop,
                          sigma_in=scale_in,
                          rho=reservoir_g,
                          alpha=reservoir_alpha,
                          feedback_connection=fb_con)

    # Training Condition
    for _ in range(N_trials_training):
        trial(ArmModel=arms,
              ReservoirModel=reservoir,
              scale_out=scale_out,
              training=True,
              do_reset=reset_after_epoch,
              arm=moving_arm,
              noise=noise)

    # Test Condition
    z_out, t_out = [], []
    for _ in range(N_trials_test):
        pred, tar = trial(
            ArmModel=arms,
            ReservoirModel=reservoir,
            scale_out=scale_out,
            training=False,
            do_reset=reset_after_epoch,
            arm=moving_arm,
        )
        z_out += list(pred)
        t_out += list(tar)

    z_out = np.array(z_out)
    t_out = np.array(t_out)

    mse = ((t_out - z_out) ** 2).mean()

    if do_plot:
        fig, ax = plt.subplots()
        ax.plot(np.array(z_out), color='r', marker=".", markersize=2, linestyle='None', alpha=0.5)
        ax.plot(np.array(t_out), color='b', alpha=0.5)
        ax.text(0.2, 0.8, f'MSE={mse:.3f}')
        plt.savefig(results_folder + "prediction_target.png")
        plt.close(fig)

    return mse


if __name__ == '__main__':
    from utils import get_element_by_interval

    simID, N_trials = int(sys.argv[1]), int(sys.argv[2])

    fb_con = bool(simID % 2)
    scale_list = [2.0, 10., 50., 100., 200., 500., 1000., 2000.]
    scale_in = get_element_by_interval(scale_list, simID, 2)

    print(f'Simulation: {simID}, Feedback: {fb_con}, Input Scaling: {scale_in}')

    run_force_training(simID=simID,
                       N_trials_training=N_trials,
                       N_trials_test=5,
                       scale_in=scale_in,
                       do_plot=True,
                       fb_con=fb_con)
