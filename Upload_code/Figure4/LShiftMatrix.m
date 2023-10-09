%% LShiftMatrix Generate a matrix that left-shifts subsystems
%   This function has two required arguments:
%       n: numer of subsystems
%       d: dimension of each subsystem
function mat = LShiftMatrix(n, d)
    mat = zeros(d^n);
    for x = 0:(d^n - 1)
        y = lshift(x, n, d);

        vec_x = zeros([d^n, 1]);
        vec_x(x + 1) = 1;
        vec_y = zeros([d^n, 1]);
        vec_y(y + 1) = 1;
        mat = mat + vec_x * vec_y.';
    end
end


%% compute y = d*x (mod d^n - 1)
function y = lshift(x, n, d)
    y = d * x;
    while y > d^n - 1
        y = y - d^n + 1;
    end
end
