"""Game logic
"""

import numpy as np
from typing import Any, Dict



class Hamiltonian:
    def __init__(self, onsite, pair):
        self.o = onsite
        self.p = pair


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

    def solve(self) -> Dict[str, Any]:
        return {"energy": -12.5, "x": np.array([1, 1, 1, 1, 1], dtype=bool)}

    def energy(self, x: np.ndarray) -> float:
        """Compute energy given a spin configuration `x`."""
        return 123
        
        
 #calculates the value for a given state       
        
def GetValueHam(H,x):
    
    return np.dot(H.o, x) + np.dot(x, np.dot(H.p,x))
    
#makes a binary list given an integer input and the length that we are looking for

def Binarize(binary,N):
    
    x = []
    
    for j in range(N):
            
            k = binary//2**(N-j-1)
            x.append(k)
            binary = binary-k*2**(N-j-1) 
    
    return x
 
#returns the maximum value of the Hamiltonian and the spin configuration

def SolveHamiltonian(H):
    
    N = len(H.o)
    
    resarray = []
    
    for i in range(2**N):
        
        #Creates a binary list of spins e.g. [0,0,1,1] for 3
        x = Binarize(i,N) 
        
        result = GetValueHam(H,x)
        
        resarray.append(result)
        
    best = [max(resarray), Binarize(resarray.index(max(resarray)),N)]   
    
    return best
    
    
#Solves it for all configs

def SolveHamiltonianAll(H):
    
    N = len(H.o)
    
    resarray = []
    
    for i in range(2**N):
        
        #Creates a binary list of spins e.g. [0,0,1,1] for 3
        x = Binarize(i,N) 
        
        result = GetValueHam(H,x)
        
        resarray.append(result)
   
    return resarray


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
    return GameState(Hamiltonian(), np.array([0, 1, 0, 0, 1], dtype=bool))


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

if __name__ == '__main__':
    play(dummy_game_state())
