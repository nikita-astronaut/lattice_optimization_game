import numpy as np
import qiskit
from qiskit.quantum_info import Pauli
from qiskit import opflow
from qiskit.opflow import PauliSumOp
from collections import OrderedDict
from qiskit import Aer
from qiskit import algorithms
from qiskit.algorithms import QAOA
from qiskit.opflow import StateFn
from qiskit.algorithms.optimizers import ADAM, COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal


def transform_interaction_to_qiskit_format(n_qubits, hamiltonian):
    r"""Generate Hamiltonian for the problem
    """

    onsite = hamiltonian.onsite
    pair = hamiltonian.pair

    def get_shift(onsite, pair):
        shift = 0.
        for onsite_term in onsite:
            _, pi = onsite_term

            shift += pi / 2.

        for pair_term in pair:
            _, _, pij = pair_term

            shift += pij / 4.
        return shift

    shift = get_shift(onsite, pair)

    def to_matrix(onsite, pair, n_qubits):
        W = np.zeros((n_qubits, n_qubits), dtype=np.float64)
        for onsite_term in onsite:
            i, pi = onsite_term
            W[i, i] += pi / 2.

        for pair_term in pair:
            i, j, pij = pair_term
            W[i, j] += pij / 8.
            W[j, i] += pij / 8

            W[i, i] += pij / 8
            W[j, j] += pij / 8

        return W

    W = to_matrix(onsite, pair, n_qubits)

    pauli_list = []

    for i in range(n_qubits):
        for j in range(n_qubits):
            if np.isclose(W[i, j], 0.0):
                continue
            x_p = np.zeros(n_qubits, dtype=bool)
            z_p = np.zeros(n_qubits, dtype=bool)
            z_p[i] = True
            z_p[j] = True
            pauli_list.append([W[i, j], Pauli((z_p, x_p))])

    pauli_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
    return PauliSumOp.from_list(pauli_list), shift

def evaluate_cost(solution, hamiltonian):
    energy = 0
    for single_term in hamiltonian.onsite:
        energy += single_term[1] * (solution[single_term[0]] == 1)

    for pair_term in hamiltonian.pair:
        energy += pair_term[2] * (solution[pair_term[0]] == 1) * (solution[pair_term[1]] == 1)

    return energy


def index_to_spin(index, n_qubits):
    return (((np.array([index]).reshape(-1, 1) & (1 << np.arange(n_qubits)))) > 0).astype(np.int64)

def bruteforce_solution(n_qubits, hamiltonian):
    energies = []
    bit_representations = []
    for idx in range(2 ** n_qubits):
        solution = index_to_spin(idx, n_qubits)[0]


        bit_representations.append(solution.copy())
        energies.append(evaluate_cost(solution, hamiltonian))

    energies = np.array(energies)
    bit_representations = np.array(bit_representations)

    return np.sort(energies), bit_representations[np.argsort(energies)]



def most_frequent_strings(state_vector, num_most_frequent):
    """Compute the most likely binary string from state vector.
    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.
    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    most_frequent_strings = [x[0] for x in sorted(state_vector.items(), \
                                                  key=lambda kv: kv[1])[-num_most_frequent:]]
    return [np.asarray([int(y) for y in (list(binary_string))]) for binary_string in most_frequent_strings]




class hamiltonian(object):
    def __init__(self, onsite, pair):
        self.onsite = onsite
        self.pair = pair
        return

n_qubits = 5

def get_random_Hamiltonian(n_qubits):
    onsite = []
    pair = []

    for i in range(n_qubits):
        onsite.append((i, np.random.uniform(-2, 2)))

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            pair.append((i, j, np.random.uniform(-2, 2)))
    return hamiltonian(onsite, pair)

ham = get_random_Hamiltonian(n_qubits)

qubit_op, offset = transform_interaction_to_qiskit_format(n_qubits, ham)
energies, bits = bruteforce_solution(n_qubits, ham)

print('ALL BRUTE FORCE SOLUTIONS')

for en, xi in zip(energies, bits):
    print('BF string:', xi, 'cost:', en)

optimizer = COBYLA()
#
#vqe = QAOA(optimizer, quantum_instance=Aer.get_backend('qasm_simulator'))#
ansatz = TwoLocal(qubit_op.num_qubits, 'ry', 'cz', reps=5, entanglement='full')
vqe = VQE(ansatz, optimizer, quantum_instance=Aer.get_backend('qasm_simulator'))

result = vqe.compute_minimum_eigenvalue(qubit_op)

x = most_frequent_strings(result.eigenstate, 4)
x = [1 - xi[::-1] for xi in x]


print('\n\n\nTESTING THE QUANTUM OUTPUT')
for xi in x:
    print('QC string:', xi, 'cost:', evaluate_cost(xi, ham))
