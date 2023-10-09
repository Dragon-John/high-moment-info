function [overhead, JD] = Method_3(k, N_noise, noise_level)
% This function corresponding to observable shift method proposed in this
% paper.
%%%%%%% returns %%%%%%%%%%%
% overhead: the optimal sampling cost given by SDP
% JD is the Choi matrix of the target channel
%%%%%%% parameters %%%%%%%%
% k: refers to the moment, i.e., Tr[rho^k]
% N_noise: is the noise model, which can be 'DE', 'AD', 'DP' which refers to
% depolarizing channel, aplitude dampling channel and dephasing channel
% respectively.
% noise_level: describe how noisy is the channel
%% define some basics and noise
X = [0 1;1 0];
Z = [1 0;0 -1];
Y = [0 1i;-1i 0];
I = eye(2);
Identity = eye(2^k);


%% define channels
perm = [];
    for i = 1:k
        perm(end+1) = 2*i-1;
    end

    for i = 1:k
        perm(end+1) = 2*i;
    end

if N_noise == 'DE'
    JN_1 = DepolarizingChannel(2,1-noise_level)*eye(4);
    JN = tensor(JN_1, JN_1);
    for i = 1:k-2
        JN = tensor(JN,JN_1);
    end

    JN = PermuteSystems(JN, perm);
end

if N_noise == 'DP'
    JN_1 = DephasingChannel(2,1-noise_level)*eye(4);
    JN = tensor(JN_1, JN_1);
    for i = 1:k-2
        JN = tensor(JN,JN_1);
    end

    JN = PermuteSystems(JN, perm);
end

if N_noise == 'AD'
    k0 = [1 0; 0 sqrt(1-noise_level)];
    k1 = [0 sqrt(noise_level); 0 0];
    JN_1 = kron(I, k0) * MaxEntangled(2,1)*MaxEntangled(2,1)'*2 * kron(I, k0') + kron(I, k1) * MaxEntangled(2,1)*MaxEntangled(2,1)'*2 * kron(I, k1');
    JN = tensor(JN_1, JN_1);
    for i = 1:k-2
        JN = tensor(JN,JN_1);
    end
    JN = PermuteSystems(JN, perm);
end

%% Define Obs
S = SwapGenerator(k);
H = 0.5*(S+S.');

%% The SDP
cvx_begin sdp quiet
    variable JD(4^k,4^k) hermitian % choi matrix of the positive part of the decoding map
    variable p
    variable t

    JF = PartialTrace(tensor(PartialTranspose(JN, 2),Identity)*tensor(Identity,JD), 2,[2^k,2^k,2^k]);
    cost = p;

    minimize cost
    subject to
        JD >= 0; PartialTrace(JD,2) == p*eye(2^k);
        ApplyMap(H, PermuteSystems(JF.', [2, 1])) == H + t*Identity;

cvx_end

%% Results
overhead = cost;
JD = JD;