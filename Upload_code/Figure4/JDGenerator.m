function [JD, t] = JDGenerator(noise_level, num_qubits)
    assert(0 <= noise_level & noise_level <= 1);
    noise_param = (1 - noise_level)^2;
    JD = JEpsGenerator(num_qubits) / noise_param;
    t = (1 - noise_param) / ((2^num_qubits) * noise_param);
end


function JEps = JEpsGenerator(num_qubits)
    dim = 2^num_qubits;

    id = tensor(eye(dim) / dim, eye(dim) / dim) * dim;
    SWAP = LShiftMatrix(2, 2^num_qubits);
    
    JEps = tensor(id, id) + tensor(SWAP - id, SWAP - id) / (dim^2 - 1);
end