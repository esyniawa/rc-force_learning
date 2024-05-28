import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.reservoir import RCNetwork
from utils import cumulative_sum, safe_save


def RCTraining(ArmModel: PlanarArms,
               ReservoirModel: RCNetwork,
               N_trials: int,
               simID: int,
               arm: str | None,
               min_movement_time: int = 50,
               max_movement_time: int = 250,
               learn_delta: int = 5,
               noise: float = 0.0,
               do_plot: bool = False):

    if arm is None:
        arm = np.random.choice(['left', 'right'])

    trajectory_folder = "trajectories/"
    # Training
    print("Motor babbling")
    for trial in range(N_trials):

        ArmModel.move_randomly(arm=arm,
                               t_min=min_movement_time,
                               t_max=max_movement_time,
                               t_wait=learn_delta+1,
                               trajectory_save_name=trajectory_folder + f"sim_{simID}/run_{trial}")

        if arm == 'right':
            input_gradient = np.array(ArmModel.trajectory_gradient_right)
            target_gradient = np.array(ArmModel.gradient_end_effector_right)
        else:
            input_gradient = np.array(ArmModel.gradient_end_effector_left)
            target_gradient = np.array(ArmModel.gradient_end_effector_left)

        if noise > 0.0:
            input_gradient += noise * np.random.uniform(-1, 1, size=input_gradient.shape)
            target_gradient += noise * np.random.uniform(-1, 1, size=target_gradient.shape)

        if trial == 0:
            inputs = input_gradient[:-learn_delta+1]
            targets = cumulative_sum(target_gradient, n=learn_delta, axis=1)
        else:
            inputs = np.concatenate((inputs, input_gradient[:-learn_delta+1]), axis=0)
            targets = np.concatenate((targets, cumulative_sum(target_gradient, n=learn_delta, axis=1)), axis=0)

        # reset trajectories
        ArmModel.clear()

    # train reservoir based on input and target
    print("Train Reservoir")
    ReservoirModel.train_target(data_in=inputs, data_target=targets)

    # Testing
    print("Test Reservoir")
    ArmModel.move_randomly(arm=arm, t_min=min_movement_time, t_max=max_movement_time, t_wait=learn_delta+1)

    if arm == 'right':
        input_gradient = ArmModel.trajectory_gradient_right
        target_gradient = ArmModel.gradient_end_effector_right
    else:
        input_gradient = ArmModel.trajectory_gradient_left
        target_gradient = ArmModel.gradient_end_effector_left

    # ReservoirModel.advance_r_state(input_gradient[0])
    prediction = ReservoirModel.predict_target(data_in=np.array(input_gradient))
    target_test = cumulative_sum(np.array(target_gradient), n=learn_delta, axis=1)

    mse = ((target_test - prediction) ** 2).mean()

    results_folder = "results/"
    if do_plot:

        traj_prediction = np.concatenate((prediction,
                                          np.zeros((len(ArmModel.end_effector_right) - prediction.shape[0], 2))),
                                         axis=0)
        if arm == 'right':
            ArmModel.plot_trajectory(dynamic_points=traj_prediction + np.array(ArmModel.end_effector_right),
                                     save_name=results_folder + f"sim_{simID}/prediction_trajectory.gif")
        else:
            ArmModel.plot_trajectory(dynamic_points=traj_prediction + np.array(ArmModel.end_effector_left),
                                     save_name=results_folder + f"sim_{simID}/prediction_trajectory.gif")

        fig, ax = plt.subplots()
        ax.plot(prediction, color='r')
        ax.plot(target_test, color='b')
        plt.savefig(results_folder + f"sim_{simID}/prediction_target.png")
        plt.close(fig)

    else:
        print(f'Test MSE = {mse:.4f}')

        # save
        safe_save(results_folder + f"sim_{simID}/mse.npy", mse)
        safe_save(results_folder + f"sim_{simID}/w_in.npy", reservoir.W_in)
        safe_save(results_folder + f"sim_{simID}/w_out.npy", reservoir.W_out)
        safe_save(results_folder + f"sim_{simID}/w_rec.npy", reservoir.A)

    return ReservoirModel


if __name__ == '__main__':

    simID, N_trials = int(sys.argv[1]), int(sys.argv[2])
    moving_arm = 'right'

    arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)
    reservoir = RCNetwork(dim_system=2, dim_reservoir=1000)

    # run training
    RCTraining(ArmModel=arms,
               ReservoirModel=reservoir,
               N_trials=N_trials,
               simID=simID,
               noise=0.01,
               arm=moving_arm,
               do_plot=True)
