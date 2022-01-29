"""Game logic
"""

import numpy as np
import scipy
import scipy.sparse
from typing import Any, Dict


class Hamiltonian:
    def __init__(self, n_qubits, onsite, pair):
        self.n_qubits = n_qubits
        self.onsite = onsite
        self.pair = pair
        return

    def _pretty_graph(self) -> str:
        return r"""
             ┌────────────┐
             │     {0}      │
        ┌────┴─────┬────┬─┘
        │          │ {2}  │
        │    {1}     │    │
        │          ├────┴──────┐
        └──┬───────┤     {3}     │
           │  {4}    ├───────────┘
           └───────┘
        """

    def pretty_graph(self, x: np.ndarray) -> str:
        x = tuple(x.astype(int).tolist())
        return self._pretty_graph().format(*x)
    

    def solve(self) -> List[Tuple[float, np.ndarray]]:
        def index_to_spin(index, n_qubits):
            return (((np.array([index]).reshape(-1, 1) & (1 << np.arange(n_qubits)))) > 0).astype(np.int64)

        energies = []
        bit_representations = []
        for idx in range(2 ** n_qubits):
            solution = index_to_spin(idx, n_qubits)[0]

        bit_representations.append(solution.copy())
        energies.append(self.energy(solution))

        energies = np.array(energies)
        bit_representations = np.array(bit_representations)

        return [(x, y) for x, y in zip(np.sort(energies), bit_representations[np.argsort(energies)])]


    def energy(self, x: np.ndarray) -> float::
        energy = 0
        for single_term in self.onsite:
            energy += single_term[1] * (x[single_term[0]] == 1)
        
        for pair_term in self.pair:
            energy += pair_term[2] * (x[pair_term[0]] == 1) * (x[pair_term[1]] == 1)
        
        return energy


def _array_to_int(xs: np.ndarray) -> int:
    r = 0
    k = 0
    for i in xs[::-1]:
        assert i == 0 or i == 1
        r |= i << k
        k += 1
    return r


def _int_to_array(r: int, n: int) -> np.ndarray:
    xs = np.zeros(n, dtype=bool)
    for i in range(n):
        xs[n - 1 - i] = (r >> i) & 1
    return xs


class ExampleHamiltonian01(Hamiltonian):
    def __init__(self):
        self.n = 4
        self.dimension = 2 ** self.n
        self.onsite = np.random.rand(self.dimension) - 0.5
        self.pair = scipy.sparse.random(self.dimension, self.dimension, density=0.5)
        self.pair.data -= 0.5
        self.pair = 0.5 * (self.pair + self.pair.T)

    def _pretty_graph(self) -> str:
        return """\
┌──────────┐
│    {0}     │
│          ├────────────┐
│          │      {1}     │
│          │            │
│   ┌──────┴─────┐      │
│   │            │      │
│   │     {2}      ├──────┤
│   ├────────────┘      │
└───┤                   │
    │          {3}        │
    └───────────────────┘"""

    def energy(self, x) -> float:
        x = _array_to_int(x)
        v = np.zeros(self.dimension, dtype=float)
        v[x] = 1
        y = self.pair @ v + self.onsite * v
        return np.dot(v, y)


class GameState:
    def __init__(self, hamiltonian: Hamiltonian, x: np.ndarray):
        self.hamiltonian = hamiltonian
        self.x = x

    def select(self, index: int):
        self.x[index] ^= 1  # Flip index'th spin
        pass


def display_current_state(state):
    print(state.hamiltonian.pretty_graph(state.x))
    print("Energy: {}".format(state.hamiltonian.energy(state.x)))


class SelectCommand:
    def __init__(self, index):
        self.index = index


class SolveAllCommand:
    def __init__(self):
        pass


class SolveSubsetCommand:
    def __init__(self, indices):
        self.indices = indices


def dummy_game_state():
    return GameState(ExampleHamiltonian01(), np.array([0, 1, 0, 1], dtype=bool))


def parse_command(s):
    first, *rest = s.split(" ")
    if first == "solve":
        if rest == ["all"]:
            return SolveAllCommand()
        else:
            indices = sum(map(lambda r: r.split(","), rest), [])
            indices = list(map(lambda r: int(r), filter(lambda r: r != "", indices)))
            return SolveSubsetCommand(indices)
    else:
        assert len(rest) == 0
        index = int(first)
        return SelectCommand(index)


def play(state: GameState):
    while True:
        display_current_state(state)
        try:
            command = parse_command(input("Specify the index of a component to select/deselect:"))
        except EOFError as e:
            print()
            break
        print(command)
        if isinstance(command, SelectCommand):
            state.select(command.index)
        elif isinstance(command, SolveAllCommand):
            solution = state.hamiltonian.solve()
            print("IonQ obtained the following solution:", solution["x"].astype(int))
            print("  it has energy:", solution["energy"])
        elif isinstance(command, SolveSubsetCommand):
            pass
        else:
            assert False


if __name__ == "__main__":
    play(dummy_game_state())
