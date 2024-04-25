%% Test 1: Simple Linear Model
rng(42); % For reproducibility
n = 100; % Number of samples
X = [ones(n, 1), randn(n, 1)]; % Feature matrix with bias term
true_theta = [2; -1.5]; % True parameter vector [b; m]
y = X * true_theta + 0.5 * randn(n, 1); % True output with noise

% Set regularization parameter
lambda = 1;

% Run QRD-RLS algorithm
param_dim = numel(true_theta);
R = 1d-3 * eye(param_dim);
U = zeros(param_dim,1);

for i=1:n
    [estimated_theta,R,U,Q,e] = QRD_RLS_Iteration(R,U,X(i,:),y(i),lambda);
end

% Display true and estimated parameter vectors
disp('True parameter vector:');
disp(true_theta');
disp('Estimated parameter vector:');
disp(estimated_theta');



%% Test 2: Polynomial Model
clear;
rng(42); % For reproducibility
n = 100; % Number of samples
X = [ones(n, 1), randn(n, 1), randn(n, 1).^2]; % Feature matrix with bias term and quadratic term
true_theta = [1; 0.5; -2]; % True parameter vector [a3; a2; a1]
y = X * true_theta + 0.5 * randn(n, 1); % True output with noise

% Set regularization parameter
lambda = 0.99;

% Run QRD-RLS algorithm
param_dim = numel(true_theta);
R = 1d-3 * eye(param_dim);
U = zeros(param_dim,1);


for i=1:n
    [estimated_theta,R,U,Q,e] = QRD_RLS_Iteration(R,U,X(i,:),y(i),lambda);
end
% Display true and estimated parameter vectors
disp('True parameter vector:');
disp(true_theta');
disp('Estimated parameter vector:');
disp(estimated_theta');



%% Test 3: Underdetermined System
clear;
rng(42); % For reproducibility
n = 30; % Number of samples
m = 3; % Number of features (more than equations)
X = [ones(n, 1), randn(n, m-1)]; % Feature matrix with bias term
true_theta = [2; -1.5; 1.2]; % True parameter vector
y = X * true_theta + 0.5 * randn(n, 1); % True output with noise

% Set regularization parameter
lambda = 1;

% Run QRD-RLS algorithm
param_dim = numel(true_theta);
R = 1d-3 * eye(param_dim);
U = zeros(param_dim,1);


for i=1:n
    [estimated_theta,R,U,Q,e] = QRD_RLS_Iteration(R,U,X(i,:),y(i),lambda);
end

% Display true and estimated parameter vectors
disp('True parameter vector:');
disp(true_theta');
disp('Estimated parameter vector:');
disp(estimated_theta');

