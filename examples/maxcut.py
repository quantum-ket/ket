# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ket import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class G:
    """Graph class to represent a graph with nodes and edges."""

    def __init__(self, n_nodes, edges, idx=None):
        """Graph structure class with iterable representation for edges"""
        self.n_nodes = n_nodes
        self.edges = edges
        self.idx = idx

    def __iter__(self):
        """Returns an iterator over the edges of the graph"""
        return G(self.n_nodes, self.edges, -1)

    def __next__(self):
        """Returns the next edge in the graph"""
        self.idx += 1
        if self.idx < len(self.edges):
            return self.edges[self.idx]
        else:
            raise StopIteration


def U_C(gamma, g, q):
    """Applies the unitary operator U_C to the qubits in `q` based on the edges and the angle `gamma`.
    For each edge (j, k) in `g`, it applies the two-qubit gate RZZ with angle `gamma` to qubits q[j] and q[k]."""
    for j, k in g:
        RZZ(gamma, q[j], q[k])


def U_B(beta, q):
    """Applies the unitary operator U_B to the qubits in `q` based on the angle `beta`.
    It applies the one-qubit gate RX with angle 2*beta to each qubit in `q`."""
    return RX(2 * beta, q)


def QAOA_base(gammas, betas, g: G, p):
    """Constructs the QAOA circuit of depth `p` with angles `gammas` and `betas` on the graph `g`.
    Returns a register of qubits after applying the circuit.

    Args:
        gammas: list of p floats, angles for U_C operators.
        betas: list of p floats, angles for U_B operators.
        g: G object representing the graph on which the QAOA algorithm is performed.
        p: integer, depth of the QAOA circuit.
    """
    q = H(quant(g.n_nodes))  # create an equal superposition of all states
    for i in range(p):
        U_C(gammas[i], g, q)
        U_B(betas[i], q)
    return q


def expval(d):
    """Calculates the expected value for a given state.

    Args:
        d (object): The state object.

    Returns:
        float: The expected value.
    """
    val = 0.0
    for state, amp in zip(d.states, d.amplitudes):
        # If the state has different values for qubits 1 and 2,
        # subtract the absolute square of the amplitude from the value
        # Otherwise, add the absolute square of the amplitude to the value
        if bool(state & 1) != bool(state & 2):
            val -= abs(amp)**2
        else:
            val += abs(amp)**2
    return val


def QAOA(gammas, betas, g: G, p):
    """Implements the QAOA algorithm.

    Args:
        gammas (list): The list of gamma parameters.
        betas (list): The list of beta parameters.
        g (G): The graph object.
        p (int): The number of layers.
    """
    # Execute the Quantum Approximate Optimization Algorithm (QAOA)
    # on the given graph with the specified parameters
    q = QAOA_base(gammas, betas, g, p)

    # Calculate the expectation value of each pair of qubits in the graph and sum them up
    ds = [dump(q[j] + q[k]) for j, k in g]
    return sum(0.5 * (1 - expval(d)) for d in ds)


def maxcut(g: G, p=1):
    """Runs the maximum cut QAOA algorithm on a given graph and prints the result.

    Args:
        g (G): A graph object containing the number of nodes and edges.
        p (int): The number of QAOA steps to run (default is 1).
    """
    # Find the maximum cut in the given graph using QAOA with the specified number of steps
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        # Minimize the negative of the QAOA value to find the maximum cut
        return -QAOA(gammas, betas, g, p)

    # Set the initial values of the parameters to 0.5
    params = [.5 for _ in range(p * 2)]
    # Use the COBYLA optimization algorithm to find the optimal parameters that maximize the QAOA value
    res = minimize(objective, params, method='COBYLA')

    # Extract the optimal parameters from the result of the optimization
    params = res.x
    # Apply the optimal parameters to the QAOA circuit to obtain the final state
    result = dump(QAOA_base(params[:p], params[p:], g, p))

    # Print the optimization result
    print(res)

    # Plot the probabilities of measuring each possible state of the qubits in the final state
    _, ax = plt.subplots()
    ax.bar([f"{i:0{g.n_nodes}b}" for i in result.states], [abs(amp)**2 for amp in result.amplitudes])
    plt.xticks(rotation=90)
    plt.show()


if __name__ == '__main__':
    # Create a graph with 4 nodes and the specified edges
    g = G(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    # Find the maximum cut in the graph using QAOA with 3 steps
    maxcut(g, 3)
