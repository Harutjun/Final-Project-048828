
SetParameters;
DefineUtilityFunctions;




%% Simualte AR(Lk) - K processes of length N

iSNR = 30; % [dB] 10*log10(Power_Signal/Power_noise)

e_t = (1/sqrt(2))*(randn(2*M + 1,N) + 1j*randn(2*M + 1,N)); % Measurments noise (Complex White Gaussian)
% v_t =(1/sqrt(2))*(randn(K,N) + j*randn(K,N)); % Process Noise 
v_t = randn(K,N); % Process Noise 

a1 = [0.872 -0.550]; % Values from the paper
a2 = [1.096 -0.870]; % Values from the paper



a = {a1,a2};

F = zeros(sum(Lks));
Gamma_mat = zeros(K,sum(Lks));
CumSumLKs = cumsum(Lks);

%%% Create the dynamics AR coefficient  matrix F
for k = 1:K
    if k ==1
        F(k:CumSumLKs(k),k:CumSumLKs(k)) = [a{k}' [eye(Lks(k)-1) zeros(Lks(k)-1,1)]']';
    else
         F(CumSumLKs(k-1)+1:CumSumLKs(k),CumSumLKs(k-1)+1:CumSumLKs(k)) = [a{k}' [eye(Lks(k)-1) zeros(Lks(k)-1,1)]']';
    end
    Gamma_mat(k,CumSumLKs(k) - Lks(k) + 1) = 1;
end


Vars_k = cellfun(@(a) iSNR2ProcessNoiseVar(iSNR,1,mean(filter(1,[1 -a] ,randn(1,1e8)).^2)),{a1,a2});

Xk = [];
v_t = randn(K,N); % Process Noise 

for k = 1:K
    Xk = cat(1,Xk,filter(sqrt(Vars_k(k)),[1, -a{k}],v_t(k,:)));
end

%%% Initialize DOA's and get  manifold matrix A
thetas = deg2rad([5, 25]);

alphas = get_alphas(thetas);
A = GetManifoldMatrix(alphas,D);

%%% Initialize Kalman Filter

P_prior =  eye(sum(Lks));
x_prior = zeros(sum(Lks),1);


%%% Filter
x_post = [];
err = [];
Y = [];
X = [];
%% Test KF
for j=1:N-Lks(1)
    xj = reshape(Xk(:,j:j+Lks(1)-1)',1,[])';
    yj = A*Gamma_mat*xj + e_t(:,j);
    Y = cat(2,Y,yj);
    X = cat(2,X,xj);
    Inov = calc_Inovation(x_prior, A*Gamma_mat,yj);
    Inov_Cov = calc_Inov_Cov(A*Gamma_mat,P_prior,eye(2*M+1));
    K_Gain = update_KalmanGain(P_prior, A*Gamma_mat, Inov_Cov);
    x_post_j = calc_x_post(x_prior, K_Gain,Inov);
    x_post = cat(2,x_post,x_post_j);
    P_post = calc_P_post(K_Gain,A*Gamma_mat,P_prior);
    err = cat(1,err,calc_err(yj,A*Gamma_mat,x_post_j'));
    x_prior = calc_x_prior(x_post_j,F);
    P_prior = calc_P_prior(P_post,diag([Vars_k(1)*ones(1,Lks(1)),Vars_k(2)*ones(1,Lks(2))]),F);

end
err = real(err);

%% DOA Estimation Test

T_2Mp1 = 1/sqrt(2) * [eye(M) zeros(M,1) 1j*eye(M) ;...
                      zeros(1,M) 2 zeros(1,M);...
                      flip(eye(M)) zeros(M,1) -1j*flip(eye(M))];   



A_T = T_2Mp1' * A;

[init_alphas,d] = meshgrid(get_alphas(deg2rad([0,40])),flip(D));
coeffGourpM_init1 = sqrt(2)*cos(d.*init_alphas);
coeffGourpM_init2 = flipud(sqrt(2)*sin(d.*init_alphas));

thetas_est = [];
Middle_idx = M + 1;
forgetting_factor = 0.8;

for i = 1:1000
    Xk = [];
    for k = 1:K
        Xk = cat(1,Xk,filter(sqrt(Vars_k(k)),[1, -a{k}],(randn(1,N+1))));
    end
    e_t = (1/sqrt(2))*(randn(2*M + 1,N) + 1j*randn(2*M + 1,N));
    Y = [];
    X = [];
    for j=2:N
        xj = reshape(fliplr(Xk(:,j:j+K-1))',1,[])';
        yj = A*Gamma_mat*xj + e_t(:,j-1);
        X = cat(2,X,xj);   
        Y = cat(2,Y,yj);
    end
    Y_T = T_2Mp1' * Y;
    R1 = 1e-8 * eye(K);
    R2 = 1e-8 * eye(K);
    est_alpha = zeros(K,1);
    for m=1:M
        
        GammaX = Gamma_mat*X;
        U1 = R1*coeffGourpM_init1(m,:)' ;
        U2 = R2*coeffGourpM_init2(m,:)' ;
        
    
        for j=1:N-1
    
            [A_Tm_est1,R1,U1,~,~] = QRD_RLS_Iteration(R1,U1,GammaX(:,j)',Y_T(Middle_idx-m,j),forgetting_factor);
    
            [A_Tm_est2,R2,U2,~,~] = QRD_RLS_Iteration(R2,U2,GammaX(:,j)',Y_T(Middle_idx+m,j),forgetting_factor);
    
        end
        est_alpha = est_alpha + (atan(real(A_Tm_est2)./real(A_Tm_est1)))./D;
        
        
    end
    est_alpha = est_alpha/M;
    thetas_est = cat(1,thetas_est,get_thetas(est_alpha'));
    [init_alphas,d] = meshgrid(est_alpha',flip(D));
    coeffGourpM_init1 = sqrt(2)*cos(d.*init_alphas);
    coeffGourpM_init2 = flipud(sqrt(2)*sin(d.*init_alphas));

        

end

thetas_est = rad2deg(thetas_est);

% Display results
histogram(thetas_est(:,1));
hold on;
histogram(thetas_est(:,2));
plot(rad2deg(thetas(1)),0,'*r','MarkerSize',10);
plot(rad2deg(thetas(2)),0,'*r','MarkerSize',10);


