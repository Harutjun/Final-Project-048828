function [Q, R] = givens_rotation(A)
% Fixed: rows i and j are both updated from their OLD values (R is upper triangular, Q*A = R).
    [m, n] = size(A);
    Q = eye(m);
    for j = 1:n
        for i = m:-1:(j+1)
            if A(i, j) ~= 0
                a_ij = A(i, j); a_jj = A(j, j);
                r = sqrt(abs(a_ij)^2 + abs(a_jj)^2);
                c = a_jj / r;  s = a_ij / r;
                G = [c', s'; -s, c];            % unitary, zeroes A(i,j)
                A([j i], :) = G * A([j i], :);
                Gf = eye(m); Gf([j i],[j i]) = G;
                Q = Gf * Q;
            end
        end
    end
    R = A;
end
