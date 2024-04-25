function [Q, R] = givens_rotation(A)
    [m, n] = size(A);
    Q = eye(m);

    for j = 1:n
        for i = m:-1:(j+1)
            if A(i, j) ~= 0
                % Calculate Givens rotation parameters
                a_ij = A(i, j);
                a_jj = A(j, j);
                c = a_jj / sqrt(a_ij*a_ij' + a_jj*a_jj');
                s = a_ij / sqrt(a_ij*a_ij' + a_jj*a_jj');

                % Update rows i and j of A
                A(i, :) = c * A(i, :) - s * A(j, :);
                A(j, :) = s' * A(i, :) + c' * A(j, :);

                % Update Q
                Q(:, [i, j]) = Q(:, [i, j]) * [c, -s; s', c'];
            end
        end
    end

    R = A;
end

