import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.reservoir import RCNetwork
from utils import cumulative_sum, safe_save


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
    print("Motor babbling")
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
        print("Train Reservoir")
        ReservoirModel.train_rls(data_in=inputs * scale_input, data_target=targets * scale_targets, do_reset=True)

        # reset trajectories
        ArmModel.clear()

    # Testing
    print("Test Reservoir")
    predictions = np.array()
    target_test = np.array()
    for _ in range(N_trials_test):
        ArmModel.move_randomly(arm=arm, t_min=min_movement_time, t_max=max_movement_time, t_wait=learn_delta+1)

        if arm == 'right':
            input_gradient = ArmModel.trajectory_gradient_right
        else:
            input_gradient = ArmModel.trajectory_gradient_left

        results_folder = "results/"

        predictions = np.concatenate((predictions,
                                      ReservoirModel.predict(data_in=np.array(input_gradient)[:-learn_delta] * scale_input)),
                                     axis=None)
        target_test = np.concatenate((target_test,
                                      ArmModel.calc_gradients(arm=arm, delta_t=learn_delta) * scale_targets),
                                     axis=None)

    mse = ((target_test - predictions) ** 2).mean()

    if do_plot:
        if arm == 'right':
            traj_prediction = np.concatenate((predictions,
                                              np.zeros((len(ArmModel.end_effector_right) - predictions.shape[0], 2))),
                                             axis=0)

            ArmModel.plot_trajectory(dynamic_points=traj_prediction + np.array(ArmModel.end_effector_right),
                                     save_name=results_folder + f"sim_{simID}/prediction_trajectory.gif")
        else:
            traj_prediction = np.concatenate((predictions,
                                              np.zeros((len(ArmModel.end_effector_left) - prediction.shape[0], 2))),
                                             axis=0)

            ArmModel.plot_trajectory(dynamic_points=traj_prediction + np.array(ArmModel.end_effector_left),
                                     save_name=results_folder + f"sim_{simID}/prediction_trajectory.gif")

        fig, ax = plt.subplots()
        ax.plot(predictions, color='r')
        ax.plot(target_test, color='b')
        plt.savefig(results_folder + f"sim_{simID}/prediction_target.png")
        plt.close(fig)

    else:
        # save
        safe_save(results_folder + f"sim_{simID}/mse.npy", mse)

    print(f'Test MSE = {mse:.4f}')

    return ReservoirModel


if __name__ == '__main__':

    simID, N_trials = int(sys.argv[1]), int(sys.argv[2])
    moving_arm = 'right'

    arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)
    reservoir = RCNetwork(dim_reservoir=1000,
                          dim_in=2, dim_out=2,
                          sigma_rec=0.2, rho=1.5, alpha=0.2)

    # run training
    RCTraining(ArmModel=arms,
               ReservoirModel=reservoir,
               N_trials_training=N_trials,
               simID=simID,
               noise=0.0,
               arm=moving_arm,
               N_trials_test=5,
               do_plot=True)
