import os.path

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
               noise: float = 0.0):

    folder = "trajectories/"
    if arm is None:
        arm = np.random.choice(['left', 'right'])

    # Training
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

        input_gradient += noise * np.random.uniform(-1, 1, size=len(input_gradient))
        target_gradient += noise * np.random.uniform(-1, 1, size=len(target_gradient))

        # train reservoir based on input and target
        ReservoirModel.train(data_in=input_gradient[:-learn_delta], data_target=target_gradient[learn_delta:])

        # reset trajectories
        ArmModel.clear()

    # Testing
    ArmModel.move_randomly(t_min=min_movement_time, t_max=max_movement_time, t_wait=learn_delta+1)

    if arm == 'right':
        input_gradient = ArmModel.trajectory_gradient_right
        target_gradient = ArmModel.gradient_end_effector_right
    else:
        input_gradient = ArmModel.gradient_end_effector_left
        target_gradient = ArmModel.gradient_end_effector_left

    ReservoirModel.predict(steps=)

    return ReservoirModel
