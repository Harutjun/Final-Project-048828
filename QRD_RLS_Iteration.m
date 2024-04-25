function [A,R,U,Q,e]= QRD_RLS_Iteration(R_prev,U_prev,x,y,lambda)

K = numel(x);

M = [sqrt(lambda)*R_prev sqrt(lambda)*U_prev ; x y];
[Q, R_block] = givens_rotation(M);
R_block = Q*M;

R = R_block(1:K,1:K);
U = R_block(1:K,K+1:end);
e = R_block(end,end);

A = R\U;

end

