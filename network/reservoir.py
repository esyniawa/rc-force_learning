import numpy as np
import networkx as nx


def generate_adjacency_matrix(dim_reservoir: int, rho: float, sigma: float, weights: str = 'normal'):
    """
    Generates a sparse adjacency matrix based on the Erdős–Rényi model

    :param dim_reservoir: The number of nodes
    :param rho: Scaling of recurrent nodes
    :param sigma: Probability for edge creation
    :return:
    """

    graph = nx.gnp_random_graph(dim_reservoir, sigma)
    graph = nx.to_numpy_array(graph)

    # Ensure random_array is of the same shape as the graph adjacency matrix.
    if weights == 'normal':
        random_array = np.random.normal(loc=0, scale=np.sqrt(1/(sigma * dim_reservoir)), size=(dim_reservoir, dim_reservoir))
    # else Uniform
    else:
        random_array = np.random.uniform(-1, 1, size=(dim_reservoir, dim_reservoir))

    # Multiply graph adjacency matrix with random values.
    rescaled = graph * random_array
    return scale_matrix(rescaled, rho)


def scale_matrix(W_rec: np.ndarray, rho: float):
    """
    :param W_rec: Adjacency matrix
    :param rho: Scaling of recurrent nodes
    :return:
    """
    eigenvalues, _ = np.linalg.eig(W_rec)
    max_eigenvalue = np.amax(eigenvalues)
    W_rec = W_rec / np.absolute(max_eigenvalue) * rho
    return W_rec


class RCNetwork:
    def __init__(self,
                 dim_reservoir: int,
                 dim_in: int,
                 dim_out: int,
                 alpha: float = 0.2,
                 rho: float = 1.2,
                 sigma_rec: float = 0.2,
                 sigma_in: float = 2.0):

        # initialize reservoir
        self.dim_out = dim_out
        self.dim_reservoir = dim_reservoir

        # initialize weights
        self.W_rec = generate_adjacency_matrix(dim_reservoir, rho, sigma_rec)
        self.W_in = sigma_in * np.random.uniform(-1, 1, size=(dim_reservoir, dim_in))
        self.W_out = np.zeros((dim_out, dim_reservoir))

        # "firing rates"
        self.r_reservoir = np.zeros(dim_reservoir)
        self.r_out = np.zeros(dim_out)

        self.P = np.eye(dim_reservoir) / alpha

    def advance_in(self, data_in: np.ndarray):
        self.r_reservoir = np.tanh(np.dot(self.W_rec, self.r_reservoir) + np.dot(self.W_in, data_in))

    def advance_out(self):
        self.r_out = np.dot(self.W_out, self.r_reservoir)

    def step(self, data_in: np.ndarray):
        self.advance_in(data_in=data_in)
        self.advance_out()

    @staticmethod
    def _rls(P, r, error):
        """
        :param P: dim_reservoir x dim_reservoir
        :param r: dim_reservoir
        :param error: dim_out
        :return: delta weights, new P
        """

        Pr = np.dot(P, r)
        rPr = np.dot(r.T, Pr).squeeze()
        c = float(1.0 / (1.0 + rPr))
        P = P - c * np.outer(Pr, Pr)

        dw = -c * np.outer(error, Pr)

        return dw, P

    def _compute_error(self, data_target: np.ndarray):
        return self.r_out - data_target

    def train_rls(self, data_in: np.ndarray, data_target: np.ndarray):
        """
        The variables have the shape (t, dim_system), t is the number of timesteps.
        :param data_in: Reservoir input
        :param data_target: Target data
        :return:
        """
        for i in range(data_in.shape[0]):
            self.step(data_in[i])

            dw, P_new = RCNetwork._rls(P=self.P, r=self.r_reservoir, error=self._compute_error(data_target[i]))
            # update readout
            self.W_out += dw
            self.P = P_new

    def predict(self, data_in: np.ndarray):
        prediction = np.zeros((data_in.shape[0], self.dim_out))
        for i in range(data_in.shape[0]):
            self.step(data_in[i])
            prediction[i] = self.r_out
        return prediction
