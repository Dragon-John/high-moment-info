## Author: @Mingrui Jing
## Date: 20230118

import numpy as np
import scipy
from scipy import sparse
import paddle
from typing import List

# tools
from QNN_QPS_EXPTOOL import *
import paddle_quantum as pq
from paddle_quantum.hamiltonian import Hamiltonian
from paddle_quantum.state import State

from openfermion import get_sparse_operator, hermitian_conjugated
from openfermion import FermionOperator
from openfermion import jordan_wigner
from openfermion import get_sparse_operator

import warnings
warnings.filterwarnings("ignore")


def get_1d_hubbard_hamiltonian(nsites: int, tunneling: float, coulomb: float, 
                               mu_up: List[float], mu_down: List[float],
                               magnetic_field = None):
    r"""Generate the Hamiltonian of a one-dimensional Hubbard chain. The framework of the qubit
        system is settled as spin-up-spin-down order. The total number of qubits is 2*nsites.

    Exp. number of sites equals to 4.
        --u--d--u--d--u--d--u--d--

    Args:
        nsites (int): number of sites.
        tunneling (float): amplitude of tunneling.
        coulomb (float): amplitude of coulomb interaction.
        mu_up (List[float]): chemical potential for spin-up particles.
        mu_down (List[float]): chemical potential for spin-down particles.
        magnetic_field (None): amplitude of magnetic field.

    Returns:
        Hamiltonian of 1d hubbard model in openfermion operator and sparse matrix.
    """

    nqubits = 2 * nsites

    # One-body (hopping) terms.
    one_body_terms = [
        op + hermitian_conjugated(op) for op in (
            FermionOperator(((i, 1), (i + 2, 0)), coefficient=-tunneling) for i in range(nqubits - 2)
        )
    ]

    # Two-body (coulomb) terms.
    two_body_terms = [
        FermionOperator(((i, 1), (i, 0), (i + 1, 1), (i + 1, 0)), coefficient=coulomb)
        for i in range(0, nqubits, 2)
    ]

    # local potential (charge-charge) terms.
    local_terms = [
        FermionOperator(((i, 1), (i, 0)), coefficient=mu_up[i//2]) + 
        FermionOperator(((i + 1, 1), (i + 1, 0)), coefficient=mu_down[i//2])
        for i in range(0, nqubits, 2)
    ]

    # derive the 1d fermionic hubbard hamiltonian
    hubbard_ferm = sum(one_body_terms) + sum(two_body_terms) + sum(local_terms)

    # J-W transformation to qubit hamiltonian
    hubbard_qubit = jordan_wigner(operator=hubbard_ferm)
    hubbard_ham = get_sparse_operator(hubbard_qubit)
    return hubbard_qubit, hubbard_ham


def occup_project_fermion(ferm_ham_mat: sparse, num_occ: int, coef: float = -1.0):
    """Derive a projected fermionic hamiltonian based on the number of electrons.

    Args:
        ferm_ham_mat (sparse): sparse matrix of original hamiltonian.
        num_elec (int): number of electrons.
    """
    # Project Fermion operator.
    dim = ferm_ham_mat.shape[0]
    diag_val = []
    diag_pos = []

    for ii in range(dim):
        ket = np.binary_repr(ii, width=int(np.log2(dim)))
        ket_a = list(map(int, ket[::2]))
        ket_b = list(map(int, ket[1::2]))

        if np.isclose(sum(ket_a)+sum(ket_b), num_occ):
            diag_val.append(coef)
            diag_pos.append(ii)

    proj_n = sparse.coo_matrix((diag_val, (diag_pos, diag_pos)), shape=(dim,dim))
    return proj_n @ ferm_ham_mat @ proj_n


def spin_project_fermion(ferm_ham_mat: sparse, spin: int = 0, coef: float = -1.0):
    """Derive a projected fermionic hamiltonian based on the unit of total spin.

    Args:
        ferm_ham_mat (sparse): sparse matrix of original hamiltonian.
        spin (int): total spin.
    """
    # Project Fermion operator.
    dim = ferm_ham_mat.shape[0]
    diag_val = []
    diag_pos = []

    for ii in range(dim):
        ket = np.binary_repr(ii, width=int(np.log2(dim)))
        ket_a = list(map(int, ket[::2]))
        ket_b = list(map(int, ket[1::2]))

        if np.isclose(sum(ket_a)-sum(ket_b), spin):
            diag_val.append(coef)
            diag_pos.append(ii)

    proj_n = sparse.coo_matrix((diag_val, (diag_pos, diag_pos)), shape=(dim,dim))
    return proj_n @ ferm_ham_mat @ proj_n


# physical properties
def get_chemical_potential(ham: Hamiltonian, gs1: State, gs2: State) -> float:
    r"""get chemical potential between states with different occupation numbers.

    Args:
        ham (Hamiltonian): Target Hamiltonian.
        gs1 (State): the first ground state.
        gs2 (State): the second ground state.

    Returns:
        float: the value of chemical potential.
    """
    eng1 = gs1.expec_val(ham)
    eng2 = gs2.expec_val(ham)

    return (eng2 - eng1).item()


def _number_op(site_idx: int):
    nferm = FermionOperator(f'{site_idx}^ {site_idx}', 1.0)
    return nferm


def charge_correlation(nsites: int, stand: int, 
                       target: int, state_density: np.ndarray,
                       err: float = 1e-5) -> float:
    r"""compute charge correlation of a given Hubbard model and state.

    Args:
        nsite (int): total number of sites.
        stand (int): standing qubit index.
        target (int): target qubit index.
        state_density (np.ndarray): input quantum state.

    Returns:
        float: the value of charge correlation.
    """
    boundary_ferm = FermionOperator(f'{2*nsites-1}^ {2*nsites-1}', err)

    n_i = _number_op(target) + _number_op(target + 1)
    n_a = _number_op(stand) + _number_op(stand + 1)
    na_ni = n_i * n_a + boundary_ferm

    # to qubitic op
    ## TODO fix the bug
    n_i_Op = qubitOperator_to_Hamiltonian(jordan_wigner(n_i+boundary_ferm)).construct_h_matrix()
    n_a_Op = qubitOperator_to_Hamiltonian(jordan_wigner(n_a+boundary_ferm)).construct_h_matrix()
    nani_Op = qubitOperator_to_Hamiltonian(jordan_wigner(na_ni)).construct_h_matrix()

    # compute correlation
    expni = np.trace(state_density @ n_i_Op)
    expna = np.trace(state_density @ n_a_Op)
    numerator = np.trace(state_density @ nani_Op) - expni * expna
    denominator = expna**2 - expni**2

    return np.real(numerator / denominator)


def spin_correlation(nsites: int, stand: int, 
                     target: int, state_density: np.ndarray,
                     err: float = 1e-5) -> float:
    r"""compute spin correlation of a given Hubbard model and state.

    Args:
        nsite (int): total number of sites.
        stand (int): standing qubit index.
        target (int): target qubit index.
        state_density (np.ndarray): input quantum state.

    Returns:
        float: the value of spin correlation.
    """
    boundary_ferm = FermionOperator(f'{2*nsites-1}^ {2*nsites-1}', err)

    S_i = _number_op(target) - _number_op(target + 1)
    S_a = _number_op(stand) - _number_op(stand + 1)
    Sa_Si = S_i * S_a + boundary_ferm

    # to qubitic op
    ## TODO fix the bug
    S_i_Op = qubitOperator_to_Hamiltonian(jordan_wigner(S_i+boundary_ferm)).construct_h_matrix()
    S_a_Op = qubitOperator_to_Hamiltonian(jordan_wigner(S_a+boundary_ferm)).construct_h_matrix()
    SaSi_Op = qubitOperator_to_Hamiltonian(jordan_wigner(Sa_Si)).construct_h_matrix()

    # compute correlation
    expSi = np.trace(state_density @ S_i_Op)
    expSa = np.trace(state_density @ S_a_Op)

    return np.real(np.trace(state_density @ SaSi_Op) - expSi * expSa)


def charge_spin_density(nsites: int, state_density: paddle.Tensor, 
                        up_site: List[int] = None, down_site: List[int] = None,
                        err: float = 1e-5) -> List:
    """Compute the charge and spin density of a given spin-based state.

    Args:
        nsites (int): number of sites.
        state_density (paddle.Tensor): spin-based state vector.
    """
    if (up_site == None) and (down_site == None):
        up_site = [i for i in range(0, 2*nsites, 2)]
        down_site = [i+1 for i in range(0, 2*nsites, 2)]
    else:
        #TODO verification
        pass

    charge_density, spin_density = [], []
    ## TODO fix the bug
    for i_site in range(nsites):
        nup = qubitOperator_to_Hamiltonian(jordan_wigner(FermionOperator(f'{up_site[i_site]}^ {up_site[i_site]}', 1.0) + FermionOperator(f'{2*nsites-1}^ {2*nsites-1}', err)))
        ndown = qubitOperator_to_Hamiltonian(jordan_wigner(FermionOperator(f'{down_site[i_site]}^ {down_site[i_site]}', 1.0) + FermionOperator(f'{2*nsites-1}^ {2*nsites-1}', err)))

        nup_Op = paddle.to_tensor(nup.construct_h_matrix())
        ndown_Op = paddle.to_tensor(ndown.construct_h_matrix())

        charge = (paddle.trace(state_density @ nup_Op).real() + paddle.trace(state_density @ nup_Op).real()).item()
        spin = (paddle.trace(state_density @ nup_Op).real() - paddle.trace(state_density @ ndown_Op).real()).item()

        charge_density.append(charge)
        spin_density.append(spin)
    
    return charge_density, spin_density


if __name__ == "__main__":

    pq.set_dtype('complex128')
    pq.set_backend("state_vector")

    # defining the Hamiltonian
    nsites = 3 # number of qubits = 2 * nsites
    U = 3.0
    J = 2.0

    # local potential (Gaussian).
    site_index = np.arange(1, nsites + 1)

    ## modify to some good values (should be good already)
    l_up, l_down = 3, 0.1
    m_up, m_down = 3, 3
    sigma_up, sigma_down = 1, 1
    epsilon_up = -l_up * np.exp(-0.5 * (site_index - m_up)**2) / sigma_up**2
    epsilon_down = -l_down * np.exp(-0.5 * (site_index - m_down)**2) / sigma_down**2

    ham_param_name = f"hubbard_nsites={nsites}_J{J}_U{U}_Gaussian(3,0.1)(3)(1)"
    task_name = "occupation"
    # create a 1*nsites-grid FH model
    hubbard_qubit, hubbard_ham = get_1d_hubbard_hamiltonian(nsites=nsites, tunneling=J, coulomb=U,
                                                            mu_up=epsilon_up.tolist(), 
                                                            mu_down=epsilon_down.tolist())
    Ham = scipy.sparse.csr_matrix.todense(hubbard_ham)

    eig_values, eig_states = scipy.linalg.eigh(Ham, subset_by_index=[0, 0])
    # print(len(eig_states))
    # print(eig_states)
    # print(sum(eig_states))
    output_state = pq.qinfo.partial_trace(eig_states@(eig_states.conj().T), 8, 8, 2)

    np.save(file='ground_state.npy', arr=output_state)