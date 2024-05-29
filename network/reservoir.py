import numpy as np
import networkx as nx


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def generate_adjacency_matrix(dim_reservoir: int, rho: float, sigma: float):
    """
    Generates a sparse adjacency matrix based on the Erdős–Rényi model

    :param dim_reservoir: The number of nodes
    :param rho: Scaling recurrent nodes
    :param sigma: Probability for edge creation
    :return:
    """

    graph = nx.gnp_random_graph(dim_reservoir, sigma)
    graph = nx.to_numpy_array(graph)

    # Ensure random_array is of the same shape as the graph adjacency matrix.
    random_array = 2 * (np.random.rand(dim_reservoir, dim_reservoir) - 0.5)

    # Multiply graph adjacency matrix with random values.
    rescaled = graph * random_array
    return scale_matrix(rescaled, rho)


def scale_matrix(W_rec: np.ndarray, rho: float):
    """
    :param W_rec:
    :param rho:
    :return:
    """
    eigenvalues, _ = np.linalg.eig(W_rec)
    max_eigenvalue = np.amax(eigenvalues)
    W_rec = W_rec / np.absolute(max_eigenvalue) * rho
    return W_rec


class RCNetwork:
    def __init__(self, dim_system, dim_reservoir, rho: float = 1.1, sigma: float = 0.1):

        # initialize reservoir
        self.dim_system = dim_system
        self.dim_reservoir = dim_reservoir

        # initialize weights
        self.W_rec = generate_adjacency_matrix(dim_reservoir, rho, sigma)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system) - .5)
        self.W_out = np.zeros((dim_system, dim_reservoir))

        # initialize "firing rate" in reservoir
        self.r_state = np.zeros(dim_reservoir)

    def advance_r_state(self, u):
        self.r_state = sigmoid(np.dot(self.W_rec, self.r_state) + np.dot(self.W_in, u))
        return self.r_state

    def v(self):
        return np.dot(self.W_out, self.r_state)

    @staticmethod
    def _linear_regression(R, trajectory, beta=0.0001):
        Rt = np.transpose(R)
        inverse_part = np.linalg.inv(np.dot(R, Rt) + beta * np.identity(R.shape[0]))
        return np.dot(np.dot(trajectory.T, Rt), inverse_part)

    def train_target_regression(self, data_in: np.ndarray, data_target: np.ndarray):
        # data_in has shape (n, dim_system), n is the number of timesteps
        R = np.zeros((self.dim_reservoir, data_in.shape[0]))
        for i in range(data_in.shape[0]):
            self.advance_r_state(data_in[i])
            R[:, i] = self.r_state

        self.W_out = RCNetwork._linear_regression(R, data_target)

    def predict_target(self, data_in: np.ndarray):
        prediction = np.zeros((data_in.shape[0], self.dim_system))
        for i in range(data_in.shape[0]):
            self.advance_r_state(data_in[i])
            prediction[i] = self.v()
        return prediction
