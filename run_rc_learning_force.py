import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.reservoir import RCNetwork
from utils import cumulative_sum, safe_save

from pybads.bads import BADS
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
          min_movement_time: int = 180,
          max_movement_time: int = 200,
          t_wait: int | None = None,
          learn_delta: int = 10,
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
        input_gradients = np.array(ArmModel.trajectory_gradient_right)
        end_effectors = np.array(ArmModel.end_effector_right[-len(input_gradients):])
    else:
        input_gradients = np.array(ArmModel.trajectory_gradient_left)
        end_effectors = np.array(ArmModel.end_effector_left[-len(input_gradients):])

    if noise > 0.0:
        input_gradients += noise * np.random.uniform(-1, 1, size=input_gradients.shape)

    targets = ArmModel.calc_gradients(array=end_effectors, delta_t=learn_delta, keep_dim=True) * scale_out

    # train reservoir based on input and target
    prediction = ReservoirModel.run(data_in=input_gradients,
                                    data_target=targets,
                                    rls_training=training,
                                    do_reset=do_reset)
    # reset trajectories
    if training:
        ArmModel.clear()
    else:
        ArmModel.clear_gradients()

    return prediction, targets


def run_force_training(simID: int,
                       N_trials_training: int,
                       N_trials_test: int,
                       scale_in: float = 1.0,
                       scale_out: float = 0.01,
                       noise: float = 0.0,
                       reset_after_epoch: bool = True,
                       moving_arm: str | None = 'right',
                       do_plot: bool = False):

    # save results in...
    results_folder = f'results/run_{simID}/'

    arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)

    reservoir = RCNetwork(dim_reservoir=1000,
                          dim_in=2,
                          dim_out=2,
                          sigma_rec=0.1,
                          sigma_in=scale_in,
                          rho=1.5,
                          alpha=0.1)

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
        if moving_arm == 'right':
            arms.plot_trajectory(dynamic_points=z_out/scale_out + np.array(arms.end_effector_right),
                                 save_name=results_folder + "prediction_trajectory.gif")
        else:
            arms.plot_trajectory(dynamic_points=z_out/scale_out + np.array(arms.end_effector_left),
                                 save_name=results_folder + "prediction_trajectory.gif")

        fig, ax = plt.subplots()
        ax.plot(np.array(z_out), color='r', marker=".", markersize=2, linestyle='None', alpha=0.5)
        ax.plot(np.array(t_out), color='b', alpha=0.5)
        ax.text(0.9, 0.9, f'MSE={mse:.3f}')
        plt.savefig(results_folder + "prediction_target.png")
        plt.close(fig)

    return mse


if __name__ == '__main__':

    simID, N_trials = int(sys.argv[1]), int(sys.argv[2])

    run_force_training(simID=simID,
                       N_trials_training=N_trials,
                       N_trials_test=20,
                       do_plot=True)
