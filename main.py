import os.path

import matplotlib.pyplot as plt
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.reservoir import RCNetwork


def RCTraining(ArmModel: PlanarArms,
               ReservoirModel: RCNetwork,
               N_trials: int,
               arm: str | None,
               min_movement_time: int = 50,
               max_movement_time: int = 250,
               learn_delta: int = 10,
               noise: float = 0.0,
               do_plot: bool = False):

    if arm is None:
        arm = np.random.choice(['left', 'right'])

    folder = "trajectories/"
    # Training
    print("Motor babbling")
    for trial in range(N_trials):

        ArmModel.move_randomly(arm=arm,
                               t_min=min_movement_time,
                               t_max=max_movement_time,
                               t_wait=learn_delta+1,
                               trajectory_save_name=folder + f"run_{trial}")

        if arm == 'right':
            input_gradient = ArmModel.trajectory_gradient_right
            target_gradient = ArmModel.gradient_end_effector_right
        else:
            input_gradient = ArmModel.gradient_end_effector_left
            target_gradient = ArmModel.gradient_end_effector_left

        if noise > 0.0:
            for i in range(len(input_gradient)):
                input_gradient[i] += noise * np.random.uniform(-1, 1, size=2)
                target_gradient[i] += noise * np.random.uniform(-1, 1, size=2)

        if trial == 0:
            inputs = input_gradient[:-learn_delta]
            targets = target_gradient[learn_delta:]
        else:
            inputs += input_gradient[:-learn_delta]
            targets += target_gradient[learn_delta:]

        # reset trajectories
        ArmModel.clear()

    # train reservoir based on input and target
    print("Train Reservoir")
    ReservoirModel.train_target(data_in=np.array(inputs), data_target=np.array(targets))

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
    prediction = ReservoirModel.predict_target(data_in=np.array(input_gradient[:-learn_delta]))

    if do_plot:
        if arm == ' right':
            ArmModel.plot_trajectory(points=prediction + np.array(ArmModel.end_effector_right))
        else:
            ArmModel.plot_trajectory(points=prediction + np.array(ArmModel.end_effector_left))

        fig, ax = plt.subplots()
        ax.plot(prediction, color='r')
        ax.plot(target_gradient[learn_delta:])
        plt.show()
        plt.close(fig)
    else:
        mse = ((target_gradient[:-learn_delta] - prediction[learn_delta:]) ** 2).mean(axis=0)
        print(f'Test MSE = {mse:.4f}')

    return ReservoirModel


if __name__ == '__main__':

    moving_arm = 'right'
    N_trials = 10_000

    arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)
    reservoir = RCNetwork(dim_system=2, dim_reservoir=1000)

    # run training
    RCTraining(ArmModel=arms,
               ReservoirModel=reservoir,
               N_trials=N_trials,
               noise=0.01,
               arm=moving_arm,
               do_plot=False)
