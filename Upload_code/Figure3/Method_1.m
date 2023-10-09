function [overhead, JD] = Method_1(k, N_noise, noise_level)
% This function provide a protocol to decompose the inverse channel into
% linear combination of CPTP maps with optimal sampling cost by SDP.
%%%%%%% returns %%%%%%%%%%%
% overhead: the optimal sampling cost given by SDP
% JD is the Choi matrix of the target channel
%%%%%%% parameters %%%%%%%%
% k: refers to the moment, i.e., Tr[rho^k]
% N_noise: is the noise model, which can be 'DE', 'AD', which refers to
% depolarizing channel and aplitude dampling channel, respectively
% noise_level: describe how noisy is the channel

%% define some basics and noise
X = [0 1;1 0];
Z = [1 0;0 -1];
Y = [0 1i;-1i 0];
I = eye(2);
Identity = eye(2^k);

%% define channels

if N_noise == 'DE'
    JN = DepolarizingChannel(2,1-noise_level)*eye(4);
end

if N_noise == 'AD'
    k0 = [1 0; 0 sqrt(1-noise_level)];
    k1 = [0 sqrt(noise_level); 0 0];
    JN = kron(I, k0) * MaxEntangled(2,1)*MaxEntangled(2,1)'*2 * kron(I, k0') + kron(I, k1) * MaxEntangled(2,1)*MaxEntangled(2,1)'*2 * kron(I, k1');
end

%% Define Obs
S = SwapGenerator(k);
H = 0.5*(S+S.');

%% The SDP
JI = MaxEntangled(2,1)*MaxEntangled(2,1)'; %choi matrix of the indentity map

cvx_begin sdp quiet
    variable J1(4,4) hermitian % choi matrix of the positive part of the decoding map
    variable J2(4,4) hermitian
    variable p1
    variable p2

    JD = J1 - J2;  % choi matrix of the decoding map
    JPI = PermuteSystems(kron(JI,JD),[2 3 1 4]);
    JF = PartialTrace(JPI*kron(JN.',eye(4)),1);  % choi matrix of the final channel

    t_N_inverse = p1+p2; % cost function (related to the sampling cost)
    minimize t_N_inverse
    subject to
        J1 >= 0; PartialTrace(J1,2) == p1*eye(2);
        J2 >= 0; PartialTrace(J2,2) == p2*eye(2);
        JF == JI;
cvx_end

overhead = t_N_inverse^k;
JD = JD;