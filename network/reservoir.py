import numpy as np
import networkx as nx


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def generate_adjacency_matrix(dim_reservoir, rho, sigma):
    graph = nx.gnp_random_graph(dim_reservoir, sigma)
    graph = nx.to_numpy_array(graph)

    # Ensure random_array is of the same shape as the graph adjacency matrix.
    random_array = 2 * (np.random.rand(dim_reservoir, dim_reservoir) - 0.5)

    # Multiply graph adjacency matrix with random values.
    rescaled = graph * random_array
    return scale_matrix(rescaled, rho)


def scale_matrix(A, rho):
    eigenvalues, _ = np.linalg.eig(A)
    max_eigenvalue = np.amax(eigenvalues)
    A = A / np.absolute(max_eigenvalue) * rho
    return A


def linear_regression(R, trajectory, beta=0.0001):
    Rt = np.transpose(R)
    inverse_part = np.linalg.inv(np.dot(R, Rt) + beta * np.identity(R.shape[0]))
    return np.dot(np.dot(trajectory.T, Rt), inverse_part)


class RCNetwork:
    def __init__(self, dim_system, dim_reservoir, rho, sigma, density):
        self.dim_system = dim_system
        self.dim_reservoir = dim_reservoir
        self.r_state = np.zeros(dim_reservoir)
        self.A = generate_adjacency_matrix(dim_reservoir, rho, sigma)
        # self.A = 0
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system) - .5)
        self.W_out = np.zeros((dim_system, dim_reservoir))

    def advance_r_state(self, u):
        self.r_state = sigmoid(np.dot(self.A, self.r_state) + np.dot(self.W_in, u))
        return self.r_state

    def v(self):
        return np.dot(self.W_out, self.r_state)

    def train_target(self, data_in: np.ndarray, data_target: np.ndarray):
        # trajectory has shape (n, dim_system), n is the number of timesteps
        R = np.zeros((self.dim_reservoir, data_in.shape[0]))
        for i in range(data_in.shape[0]):
            self.advance_r_state(data_in[i])
            R[:, i] = self.r_state

        self.W_out = linear_regression(R, data_target)

    def predict_target(self, data_in: np.ndarray):
        prediction = np.zeros((data_in.shape[0], self.dim_system))
        for i in range(data_in.shape[0]):
            self.advance_r_state(data_in[i])
            prediction[i] = self.v()
        return prediction
