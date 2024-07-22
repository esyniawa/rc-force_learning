import numpy as np
import matplotlib.pyplot as plt

from kinematics.planar_arms import PlanarArms
from network.reservoir import RCNetwork


arms = PlanarArms(init_angles_left=np.array((20, 20)), init_angles_right=np.array((20, 20)), radians=False)

reservoir = RCNetwork(dim_reservoir=200,
                      dim_in=2,
                      dim_out=2,
                      sigma_rec=0.2,
                      sigma_in=10.,
                      rho=1.2,
                      alpha=0.1,
                      feedback_connection=False)


input_theta = []
target_grad = []
out_grad = []

def trial(ArmModel: PlanarArms,
          ReservoirModel: RCNetwork,
          training: bool,
          scale_out: float,
          do_reset: bool = False,
          arm: str = 'right',
          min_movement_time: int = 60,
          max_movement_time: int = 120,
          t_wait: int | None = 20,
          learn_delta: int = 5,
          noise_sd: float = 0.0):

    if t_wait is None:
        t_wait = learn_delta + 1

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

    if noise_sd > 0.0:
        thetas += np.random.normal(0, noise_sd, size=thetas.shape)

    input_gradients = ArmModel.calc_gradients(array=thetas, delta_t=learn_delta)
    targets = ArmModel.calc_gradients(array=end_effectors, delta_t=learn_delta) * scale_out

    # train reservoir based on input and target
    prediction = ReservoirModel.run(data_in=input_gradients,
                                    data_target=targets,
                                    rls_training=training,
                                    do_reset=do_reset)
    # reset trajectories
    ArmModel.clear()

    input_theta.append(input_gradients)
    out_grad.append(prediction)
    target_grad.append(targets)


for _ in range(5):
    trial(ArmModel=arms,
          ReservoirModel=reservoir,
          scale_out=0.01,
          training=True)

fig, axs = plt.subplots(nrows=2)
axs[0].plot(np.concatenate(input_theta))
axs[1].plot(np.concatenate(out_grad), 'b')
axs[1].plot(np.concatenate(target_grad), 'r')
plt.show()

