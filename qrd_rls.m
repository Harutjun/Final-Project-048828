function [theta, P] = qrd_rls(X, y, lambda)
    % Initialize parameters
    [n, m] = size(X); % Number of samples and features
    P = lambda * eye(m); % Initial inverse correlation matrix
    theta = zeros(m, 1); % Initial parameter vector
    
    % Iterate through each sample
    for k = 1:n
        x_k = X(k, :)'; % Input vector for sample k
        y_k = y(k); % Desired output for sample k
        
        % Prediction and error
        y_hat = x_k' * theta; % Predicted output
        e = y_k - y_hat; % Error
        
        % QR decomposition
        [Q, R] = qr([P * x_k, x_k], 0);
        
        % Update gain vector
        K = P * x_k / R';
        
        % Update parameter vector and inverse correlation matrix
        theta = theta + K * e;
        P = (P - K * x_k' * P) / lambda;
    end
end