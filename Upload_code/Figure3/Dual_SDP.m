clear;
I = eye(2);
II = eye(4);
noise_level = 0.1;
%% DE Noise

JN_1 = DepolarizingChannel(2,1-noise_level)*eye(4);
JN = tensor(JN_1, JN_1);
JN = PermuteSystems(JN, [1,3,2,4]);

%% AD Noise
k0 = [1 0; 0 sqrt(1-noise_level)];
k1 = [0 sqrt(noise_level); 0 0];
JN_1 = kron(I, k0) * MaxEntangled(2,1)*MaxEntangled(2,1)'*2 * kron(I, k0') + kron(I, k1) * MaxEntangled(2,1)*MaxEntangled(2,1)'*2 * kron(I, k1');
JN = tensor(JN_1, JN_1);
JN = PermuteSystems(JN, [1,3,2,4]);

% JN_1 = RandomSuperoperator(2);
% JN = tensor(JN_1, JN_1);
% JN = PermuteSystems(JN, [1,3,2,4]);
%% Observable
S = SwapGenerator(2);
H = 0.5*(S+S.');
%% Prime SDP
cvx_begin sdp quiet
    variable JD(4^2,4^2) hermitian % choi matrix of the positive part of the decoding map
    variable p
    variable t

    JF = PartialTrace(tensor(PartialTranspose(JN, 2),eye(2^2))*tensor(eye(2^2),JD), 2,[2^2,2^2,2^2]);
    prime = p;

    minimize prime
    subject to
        JD >= 0; PartialTrace(JD,2) == p*eye(4);
        ApplyMap(H, PermuteSystems(JF.', [2, 1])) == H + t*eye(2^2);

cvx_end
%% Dual SDP
cvx_begin sdp quiet
    variable M(4,4) hermitian
    variable K(4,4) hermitian

    S_1 = tensor(K.',II,H);
    S_2 = tensor(PartialTranspose(JN, 2), II);
    % S_2 = tensor(JN.', II);
    PT = PartialTrace(S_1*S_2, 1, [4, 4, 4]);

    dual = -trace(K*H);
    maximize dual
    subject to
        trace(M) <= 1
        trace(K) == 0
        tensor(M,II) + PT >= 0
cvx_end
prime
dual