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
               number_movements_test: int,
               arm: str | None,
               min_movement_time: int = 50,
               max_movement_time: int = 250,
               learn_delta: int = 5,
               noise: float = 0.0,
               save_trajectories: bool = False,
               do_plot: bool = False):

    if arm is None:
        arm = np.random.choice(['left', 'right'])


    # Training
    print("Motor babbling")
    for trial in range(N_trials):

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

        if trial == 0:
            inputs = input_gradient[:-learn_delta]
            targets = ArmModel.calc_gradients(arm=arm, delta_t=learn_delta)
        else:
            inputs = np.concatenate((inputs, input_gradient[:-learn_delta]), axis=0)
            targets = np.concatenate((targets, ArmModel.calc_gradients(arm=arm, delta_t=learn_delta)), axis=0)

        # reset trajectories
        ArmModel.clear()

    # train reservoir based on input and target
    print("Train Reservoir")
    ReservoirModel.train_rls(data_in=inputs, data_target=targets)

    # Testing
    print("Test Reservoir")
    for _ in range(number_movements_test):
        ArmModel.move_randomly(arm=arm, t_min=min_movement_time, t_max=max_movement_time, t_wait=learn_delta+1)

    if arm == 'right':
        input_gradient = ArmModel.trajectory_gradient_right
    else:
        input_gradient = ArmModel.trajectory_gradient_left

    # ReservoirModel.advance_r_state(input_gradient[0])
    prediction = ReservoirModel.predict(data_in=np.array(input_gradient)[:-learn_delta])
    target_test = ArmModel.calc_gradients(arm=arm, delta_t=learn_delta)

    mse = ((target_test - prediction) ** 2).mean()

    results_folder = "results/"
    if do_plot:
        if arm == 'right':
            traj_prediction = np.concatenate((prediction,
                                              np.zeros((len(ArmModel.end_effector_right) - prediction.shape[0], 2))),
                                             axis=0)

            ArmModel.plot_trajectory(dynamic_points=traj_prediction + np.array(ArmModel.end_effector_right),
                                     save_name=results_folder + f"sim_{simID}/prediction_trajectory.gif")
        else:
            traj_prediction = np.concatenate((prediction,
                                              np.zeros((len(ArmModel.end_effector_left) - prediction.shape[0], 2))),
                                             axis=0)

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
        safe_save(results_folder + f"sim_{simID}/w_rec.npy", reservoir.W_rec)

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
               N_trials=N_trials,
               simID=simID,
               noise=0.0,
               arm=moving_arm,
               number_movements_test=5,
               do_plot=True)
