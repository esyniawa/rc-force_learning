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

def RCTraining(ArmModel: PlanarArms,
               ReservoirModel: RCNetwork,
               N_trials_training: int,
               N_trials_test: int,
               simID: int,
               arm: str | None,
               min_movement_time: int = 100,
               max_movement_time: int = 200,
               scale_input: float = 10.,
               scale_targets: float = 1.,
               learn_delta: int = 10,
               noise: float = 0.01,
               save_trajectories: bool = False,
               do_plot: bool = False):

    if arm is None:
        arm = np.random.choice(['left', 'right'])


    # Training
    for trial in range(N_trials_training):

        if save_trajectories:
            ArmModel.move_randomly(arm=arm,
                                   t_min=min_movement_time,
                                   t_max=max_movement_time,
                                   t_wait=learn_delta+1,
                                   trajectory_save_name=f"trajectories/sim_{simID}/run_{trial}")
        else:
            ArmModel.move_randomly(arm=arm,
                                   t_min=min_movement_time,
                                   t_max=max_movement_time,
                                   t_wait=learn_delta+1)

        if arm == 'right':
            input_gradient = np.array(ArmModel.trajectory_gradient_right)
        else:
            input_gradient = np.array(ArmModel.gradient_end_effector_left)

        if noise > 0.0:
            input_gradient += noise * np.random.uniform(-1, 1, size=input_gradient.shape)

        inputs = input_gradient[:-learn_delta]
        targets = ArmModel.calc_gradients(arm=arm, delta_t=learn_delta)

        # train reservoir based on input and target
        ReservoirModel.train_rls(data_in=inputs * scale_input, data_target=targets * scale_targets, do_reset=True)

        # reset trajectories
        ArmModel.clear()

    # Testing
    for trial in range(N_trials_test):
        ArmModel.move_randomly(arm=arm, t_min=min_movement_time, t_max=max_movement_time, t_wait=learn_delta+1)

        if arm == 'right':
            input_gradient = ArmModel.trajectory_gradient_right

        else:
            input_gradient = ArmModel.trajectory_gradient_left

        results_folder = "results/"

        # output RC
        prediction = ReservoirModel.predict(data_in=np.array(input_gradient) * scale_input)

        if trial == 0:
            predictions = prediction
        else:
            predictions = np.concatenate((predictions, prediction), axis=0)
        # reset gradients
        ArmModel.clear_gradients()

    target_tests = ArmModel.calc_gradients(arm=arm, delta_t=learn_delta, keep_dim=True) * scale_targets
    mse = ((target_tests - predictions) ** 2).mean()

    if do_plot:
        if arm == 'right':
            ArmModel.plot_trajectory(dynamic_points=predictions + np.array(ArmModel.end_effector_right),
                                     save_name=results_folder + f"sim_{simID}/prediction_trajectory.gif")
        else:
            ArmModel.plot_trajectory(dynamic_points=predictions + np.array(ArmModel.end_effector_left),
                                     save_name=results_folder + f"sim_{simID}/prediction_trajectory.gif")

        fig, ax = plt.subplots()
        ax.plot(predictions, color='r')
        ax.plot(target_tests, color='b')
        plt.savefig(results_folder + f"sim_{simID}/prediction_target.png")
        plt.close(fig)

    return mse


def fit_force_training(simID: int,
                       N_trial_training: int,
                       noise: float = 0.0,
                       moving_arm: str = 'right'):

    arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)

    def loss_function(res_params):

        dim_res, sigma_rec, rho, alpha, scale_in = res_params

        reservoir = RCNetwork(dim_reservoir=int(dim_res),
                              dim_in=2, dim_out=2,
                              sigma_rec=sigma_rec, rho=rho, alpha=alpha)

        fitting_error = RCTraining(ArmModel=arms,
                                   ReservoirModel=reservoir,
                                   N_trials_training=N_trial_training, simID=simID,
                                   noise=noise,
                                   arm=moving_arm,
                                   N_trials_test=15,
                                   scale_input=scale_in,
                                   do_plot=False)

        return fitting_error

    init_params = 1000, 0.2, 1.2, 0.2, 10.
    target = loss_function

    bads = BADS(target, np.array(init_params),
                lower_bounds=np.array((100, 0.05, 0.8, 0.01, 1.0)),
                upper_bounds=np.array((5000, 1.0, 2.0, 1.0, 500.0)))

    optimize_result = bads.optimize()
    fitted_params = optimize_result['x']

    safe_save(f'results/fit_run_{simID}/fitted_params.npy', fitted_params)

def run_force_training(simID: int,
                       N_trial_training: int,
                       noise: float = 0.0,
                       moving_arm: str = 'right'):

    arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)

    reservoir = RCNetwork(dim_reservoir=1000,
                          dim_in=2, dim_out=2,
                          sigma_rec=0.2, rho=1.4, alpha=0.1)

    RCTraining(ArmModel=arms,
               ReservoirModel=reservoir,
               N_trials_training=N_trial_training, simID=simID,
               noise=noise,
               arm=moving_arm,
               N_trials_test=15,
               scale_input=5.0,
               do_plot=True)

if __name__ == '__main__':

    simID, N_trials = int(sys.argv[1]), int(sys.argv[2])

    run_force_training(simID=simID, N_trial_training=N_trials)
