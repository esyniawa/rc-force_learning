# Reservoir Computing - RLS Force Learning

## Description
This project uses the RLS (Recursive Least Squares) force learning rule to train forward prediction based on joint angle changes, which indicate where the endeffector will be in the future. 

## Files Structure
- `run_rc_learning_force.py`: Main script to run force training simulations.
- `test_rc.py`: Test script to show that the learning rule and the reservoir definitions works.
- `network/reservoir`: Contains the `RCNetwork` class that implement a Reservoir and the learning rule to train the weights.
- `kinematics/planar_arms.py`: Contains the `PlanarArms` class with various methods for arm control and visualization.

## Usage
1. Ensure Python environment is set up.
```bash 
conda env create -n [name] --file env.yml
conda activate [name]
```
2. Run `run_rc_learning_force.py` with simulation ID and number of trials as command-line arguments. The number of trials determine the number of executed movements over which the prediction is learned.

```bash 
python run_rc_learning_force.py 1 10000
```

## `PlanarArms` Class
- `__init__`: Initialize arm with initial angles.
- `forward_kinematics`: Calculate forward kinematics.
- `inverse_kinematics`: Calculate inverse kinematics.
- `reset_all`: Reset arm to initial state.
- `set_trajectory`: Set arm trajectory.
- `change_angle`: Move arm to new arm angles.
- `change_position_straight`: Move arm to a new position in the shortest way.
- `move_to_position`: Move arm to a specific end effector position.
- `wait`: Pause arm movement for a specified number of time steps.
- `move_randomly`: Move arm randomly within specified constraints.
- `save_state`: Save arm state.
- `import_state`: Import arm state from a file.

#### Visualizations
- `plot_current_position`: Plot current arm position.
- `plot_trajectory`: Plot arm trajectory with dynamic points.

#### Additional Functions
- `calc_gradients`: Calculate gradients for arm movement.
- `calc_motor_vector`: Calculate motor vector based on initial and end positions.
- `calc_position_from_motor_vector`: Calculate position from motor vector.

#### Notes
- Ensure to specify the arm side (left or right) when using specific methods.
- Follow the provided visualisation parameters for accurate plotting.

## `RCNetwork` class

- `__init__`: Initializes the network with the specified dimensions and weights for the reservoir, input, and output layers.
- `step`: Performs a given time step of the RCNetwork.
- `reset_reservoir`: Resets the reservoir state to zeros.
- `_rls` (static method): Implements the Recursive Least Squares algorithm to update weights and the matrix P.
- `run`: Run the reservoir network on the given input and target data.
- `predict`: Predicts outputs based on input data, optionally saves reservoir activities and output weights, and resets the reservoir after the epoch if specified.
