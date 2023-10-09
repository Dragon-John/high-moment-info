import numpy as np
from typing import List, Dict, Tuple

from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['figure.dpi'] = 500

import paddle
import paddle_quantum as pq
from paddle_quantum.ansatz import Circuit
from paddle_quantum.channel import ChoiRepr
from paddle_quantum.qinfo import von_neumann_entropy, partial_trace
from paddle_quantum.state import State, random_state, bell_state
from paddle_quantum.trotter import get_1d_heisenberg_hamiltonian

dtype = 'complex128'
pq.set_backend('density_matrix')
pq.set_dtype(dtype)

import matlab
import matlab.engine

eng = matlab.engine.start_matlab()

seed = np.random.randint(0, 1000)
paddle.seed(seed)
print('seed:', seed)

# set data, noise and quantum state
num_qubits = 3
noise_param = 0.1
rho = State(np.load('ground_state.npy'))
input_state = rho.kron(rho)

expect_val = paddle.trace(rho @ rho).real().item()

plot_size, num_bins = 300, 150
sample_size, batch_size = 10000, 2048


S = np.array(eng.LShiftMatrix(matlab.double(2), matlab.double(2 ** num_qubits), nargout=1))
H = paddle.to_tensor(0.5 * (S + S.conj().T), dtype=dtype)

eigvals, eigvecs = np.linalg.eigh(H.numpy())

def __near(arr, y):
    return np.where(np.abs(arr - y) < 1e-6)[0]

idx_1, idx_n1 = __near(eigvals, 1), __near(eigvals, -1)
proj_1 = sum(np.outer(eigvecs[:, i], eigvecs[:, i].conj().T) for i in idx_1)
proj_n1 = sum(np.outer(eigvecs[:, i], eigvecs[:, i].conj().T) for i in idx_n1)

qubits_idx = list(range(num_qubits))

cir = Circuit(num_qubits * 2)
cir.generalized_depolarizing(noise_param, qubits_idx)
cir.generalized_depolarizing(noise_param, [i + num_qubits for i in qubits_idx])


noisy_state = cir(input_state)
noisy_val = paddle.trace(H @ noisy_state.data).real().item()

JD, t = eng.JDGenerator(matlab.double(noise_param), 
                        matlab.double(num_qubits), nargout=2)
JD, t = paddle.to_tensor(np.array(JD)), np.array(t).item()
cir.choi_channel(JD, list(range(2 * num_qubits)))

##
em_state = cir(input_state)
em_val = paddle.trace(H @ em_state.data).real().item() - t

##
noisy_state, em_state, H = noisy_state.numpy(), em_state.numpy(), H.numpy()

##
noisy_prob = [np.trace(proj_1 @ noisy_state).real, np.trace(proj_n1 @ noisy_state).real]
em_prob = [np.trace(proj_1 @ em_state).real, np.trace(proj_n1 @ em_state).real]

# Sampling data

def batch_avg(prob: List[float]):
    r"""
    """
    sum_prob = sum(prob)
    outcome = np.random.choice([1, -1], size=batch_size, p=prob / sum_prob)
    return sum(outcome) * sum_prob / batch_size

def sample_noisy(prob: List[float]):
    r"""
    """
    return [batch_avg(prob) for _ in range(sample_size)]

def sample_em(prob: List[float]):
    r"""
    """
    return [batch_avg(prob) - t for _ in range(sample_size)]

noisy_data = sample_noisy(noisy_prob)
em_data = sample_em(em_prob)

# Plot figure

is_density = False
noisy_data, em_data = sample_noisy(noisy_prob), sample_em(em_prob)
em_mu, em_std = norm.fit(em_data)
xmin, xmax = em_mu - 5 * em_std, em_mu + 3 * em_std
x = np.linspace(xmin, xmax, plot_size)


def plot_data(data, color, label) -> None:
    r"""
    """
    # data = data[xmin <= data <= xmax]
    _, bins, _ = plt.hist(data, bins=num_bins, range=(0.2, 0.9),
                          density=is_density, color=color, alpha=0.3)
    mu, std = np.mean(data), np.std(data)
    y = norm.pdf(x, mu, std)
    
    if is_density:
        plt.plot(x, y, 
                 color=color, label=label)
    else:
        plt.plot(x, y * len(data) * (bins[1] - bins[0]), 
                 color=color, label=label)

plot_data(noisy_data, color='#f48153', label='Sample noisy states')


plot_data(em_data, color='#83d3d4', label='Sample with protocol')

plt.axvline(x=paddle.trace(rho @ rho).real().item(), color='black', linestyle='--')


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(xmin, xmax)
plt.ylim(0)

plt.xlabel('Expected values from the sampling', fontsize=14)
plt.ylabel('Probability density' if is_density else 'Frequency', fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.savefig(fname='Hubbard_Simulation.png')
# plt.show()
