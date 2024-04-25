function A = GetManifoldMatrix(alphas,ds)

M = numel(ds);
K = numel(alphas);

[Y,X] = meshgrid(alphas,flip(ds));
A_1 = exp(1i*X.*Y);

J_M = flip(eye(M));

A = transpose([transpose(A_1) ones(1,K)'   (J_M * A_1)']);


end