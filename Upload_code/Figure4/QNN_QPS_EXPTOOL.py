## Author: @Mingrui Jing
## Date: 20230118

import numpy as np
import paddle
import random

import paddle_quantum as pq
from paddle_quantum.hamiltonian import Hamiltonian
from paddle_quantum.state import random_state, State

# from openfermion.measurements import group_into_tensor_product_basis_sets
from openfermion.ops import QubitOperator

import warnings
warnings.filterwarnings("ignore")

pq.set_dtype('complex128')
pq.set_backend("state_vector")


def p_overlap_random_state(target: State, p: float = 1.0) -> State:
    r"""Function used to create a random p-overlap state regarding a target.

    Args:
        target (State): Target state.
        p (float, optional): Overlap. Defaults to 1.0.

    Returns:
        State: The result p-overlap state.
    """
    if p == 1.0:
        return State(np.exp(random.random()*1j) * target.ket)

    target_array = target.ket.numpy()
    guide_state = random_state(target.num_qubits)
    while paddle.abs(target.bra @ guide_state.ket) == 0.99999:
        guide_state = random_state(target.num_qubits)
    
    source_array = np.concatenate((target_array, guide_state.ket.numpy()), axis=1)
    orthQ, rectR = np.linalg.qr(source_array)

    proj_array = orthQ[:, 1]
    p_overlap_array = p * target_array.reshape(2**target.num_qubits) + np.sqrt(1-p**2) * proj_array
    p_overlap_state = State(p_overlap_array)
    return p_overlap_state


def ham_shuffling(ham: Hamiltonian) -> Hamiltonian:
    r"""Generate random shuffling Hamiltonian given ham. (shuffling coefficients)

    Args:
        ham (Hamiltonian): target Hamiltonian.

    Returns:
        Hamiltonian: shuffled Hamiltonian.
    """
    pauli = ham.pauli_str
    new_coeff = np.random.random(ham.n_terms) * np.random.choice([-1.0, 1.0], ham.n_terms)
    new_ham_list = []
    for i in range(ham.n_terms):
        suffle_pauli = [new_coeff[i], pauli[i][1]]
        new_ham_list.append(suffle_pauli)
    return Hamiltonian(new_ham_list)


def of_operator_transform(ham: Hamiltonian) -> Hamiltonian:
    r"""Openfermion operator transformer.

    Args:
        ham (Hamiltonian): target Hamiltonian.

    Returns:
        Hamiltonian: The corresponding Openfermion Hamiltonian (Qubit) operator.
    """
    pauli_str = ham.pauli_str
    of_op = QubitOperator()
    for pterm_tuple in pauli_str:
        p_term_list = [pterm for pterm in pterm_tuple[1].split(',') if pterm[0] != 'I']
        of_pterm_str = ' '.join(p_term_list)
        of_op += QubitOperator(of_pterm_str, pterm_tuple[0])

    return of_op    

def tol_filter(data_list: list, tol: float = 0.1) -> list:
    r"""Filter out all data values in a list smaller than the tol.

    Args:
        data_list (list): data list.
        tol (float, optional): tolerance. Defaults to 0.1.

    Returns:
        list: resulting list of data
    """
    new_list = data_list.copy()
    for data in data_list:
        if data <= tol:
            new_list.remove(data)
    return new_list
