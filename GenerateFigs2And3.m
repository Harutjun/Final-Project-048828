if ~exist('overrideParams','var')
    K = 2; % Num of sources
    N = 30; % Num of Snapshoots per Interval
    N_Int = 200; % Num of Intervarls
    Lks = [2,2]; % order of sources AR processes
    M = 1; % size of the subarrays
    lambda = 1; % Wavelength
    D = (lambda/4) * ones(M,1); % Distances between elements in subarrays
    iSNR = 30; % [dB] 10*log10(Power_Signal/Power_noise)
    a1 = [0.872 -0.550]; % Values from the paper
    a2 = [1.096 -0.870]; % Values from the paper
    a = {a1,a2};
    figname2 = 'Fig2.png';
    figname3 = 'Fig3.png';
end
DefineUtilityFunctions;
rng(20); % For Reproducability
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


thetas = deg2rad([0, 20]);
alphas = get_alphas(thetas);
A = GetManifoldMatrix(alphas,D);

a_est1 = [];
a_est2 = [];
% Run QRD-RLS algorithm
R1 = 1e-13 * eye(Lks(1));
U1 = R1 * [0.772 ;-0.450];

R2 = 1e-13 * eye(Lks(1));
U2 = R2 * [0.96; -0.77];
err = [];
thetas_est = [];
Middle_idx = M + 1;
forgetting_factor = 0.995;

[init_alphas,d] = meshgrid(get_alphas(deg2rad([5,25])),flip(D));
coeffGourpM_init1 = sqrt(2)*cos(d.*init_alphas);
coeffGourpM_init2 = flipud(sqrt(2)*sin(d.*init_alphas));
N_Tot = N * N_Int;
X1 = filter(sqrt(Vars_k(1)),[1, -a{1}],(1/sqrt(2))*(randn(1,N_Tot) + 1j*randn(1,N_Tot)));
X2 = filter(sqrt(Vars_k(2)),[1, -a{2}],(1/sqrt(2))*(randn(1,N_Tot) + 1j*randn(1,N_Tot)));

for i = 1:N_Int
        P_prior =  exp(1j*pi/4)*eye(sum(Lks));
        x_prior = zeros(sum(Lks),1);
        startIdx = i*(N-1)+1;
        stopIdx = startIdx + N -1;
        Xk = cat(1,X1(startIdx:stopIdx),X2(startIdx:stopIdx));
        Y = [];
        %%% Filter
        x_post = [];
        e_t = (1/sqrt(2))*(randn(2*M + 1,N) + 1j*randn(2*M + 1,N)); % Measurments noise (Complex White Gaussian)
    for j=Lks(1):N
        xj = reshape(Xk(:,j-Lks(1)+1:j)',1,[])';
        yj = A*Gamma_mat*xj + e_t(:,j);
        Y = cat(2,Y,yj);
        Inov = calc_Inovation(x_prior, A*Gamma_mat,yj);
        Inov_Cov = calc_Inov_Cov(A*Gamma_mat,P_prior,exp(1j*pi/4)*(eye(2*M+1)));
        K_Gain = update_KalmanGain(P_prior, A*Gamma_mat, Inov_Cov);
        x_post_j = calc_x_post(x_prior, K_Gain,Inov);
        x_post = cat(2,x_post,x_post_j);
        P_post = calc_P_post(K_Gain,A*Gamma_mat,P_prior);
        err = cat(1,err,calc_err(yj,A*Gamma_mat,x_post_j'));
        x_prior = calc_x_prior(x_post_j,F);
        P_prior = calc_P_prior(P_post,diag([Vars_k(1)*ones(1,Lks(1)),Vars_k(2)*ones(1,Lks(2))]),F);
    end
    Z1 = [real(x_post(1,2:end)), imag(x_post(1,2:end))];
    B1 = [real(x_post(1,1:end-1)), imag(x_post(1,1:end-1))];

    Z2 = [real(x_post(3,2:end)), imag(x_post(3,2:end))];
    B2 = [real(x_post(3,1:end-1)), imag(x_post(3,1:end-1))];
    for j=Lks(2):numel(Z1)
        [aj_est1,R1,U1,~,~] = QRD_RLS_Iteration(R1,U1,B1(j-Lks(1)+1:j),Z1(j),0.995);
        [aj_est2,R2,U2,~,~] = QRD_RLS_Iteration(R2,U2,B2(j-Lks(1)+1:j),Z2(j),0.995);
    end
    a_est1 = cat(2,a_est1,aj_est1);
    a_est2 = cat(2,a_est2,aj_est2);

    T_2Mp1 = 1/sqrt(2) * [eye(M) zeros(M,1) 1j*eye(M) ;...
                      zeros(1,M) 2 zeros(1,M);...
                      flip(eye(M)) zeros(M,1) -1j*flip(eye(M))];   
    Y_T = T_2Mp1' * Y;

    est_alpha = zeros(K,1);
    R1_DOA = 1e-3 * eye(K);
    R2_DOA = 1e-3 * eye(K);
    for m=1:M
        
        GammaX = Gamma_mat*x_post;
        U1_DOA = R1_DOA*transpose(coeffGourpM_init1(m,:)) ;
        U2_DOA = R2_DOA*transpose(coeffGourpM_init2(m,:)) ;
    
    
        for j=1:size(Y_T,2)
    
            [A_Tm_est1,R1_DOA,U1_DOA,~,~] = QRD_RLS_Iteration(R1_DOA,U1_DOA,transpose(GammaX(:,j)),Y_T(Middle_idx-m,j),forgetting_factor);
    
            [A_Tm_est2,R2_DOA,U2_DOA,~,~] = QRD_RLS_Iteration(R2_DOA,U2_DOA,transpose(GammaX(:,j)),Y_T(Middle_idx+m,j),forgetting_factor);
            
        end
        est_alpha = est_alpha + real(atan((A_Tm_est2)./(A_Tm_est1)))./D(m);    
    end 
    
    est_alpha = est_alpha/M;
    thetas_est = cat(1,thetas_est,get_thetas(est_alpha'));
    [init_alphas,d] = meshgrid(est_alpha',flip(D));
    coeffGourpM_init1 = sqrt(2)*cos(d.*init_alphas);
    coeffGourpM_init2 = flipud(sqrt(2)*sin(d.*init_alphas));
end
    






a_est1 = a_est1';
a_est2 = a_est2';
% Display results
f2 = figure;
sgtitle('Figure 2')
subplot 121;
histogram(real(a_est1(:,1)));
hold on;
ylabel('Histogram');
for l = 2:size(a_est1,2)
    histogram(real(a_est1(:,l)));
end
plot(a{1},0,'*r','MarkerSize',10);
subtitle(sprintf('AR Coefficient Of The First Source \n (a)'));
legend('Estimated','Estimated','True')
subplot 122;
histogram(real(a_est2(:,1)));
hold on;

for l = 2:size(a_est2,2)
    histogram(real(a_est2(:,l)));
end
plot(a{2},0,'*r','MarkerSize',10);
ylabel('Histogram');
subtitle(sprintf('AR Coefficient Of The Second Source \n (b)'));
legend('Estimated','Estimated','True')
thetas_est = rad2deg(thetas_est)';
% Display results

f3 = figure;
hold on;
for l = 1:size(thetas_est,1)
    histogram(thetas_est(l,:));
end
plot(rad2deg(thetas),zeros(size(thetas)),'r*','MarkerSize',15)
sgtitle('Figure 3');
legend('Estimated','Estimated','True')
xlabel('DOA [Degree]')
ylabel('Histogram');

saveas(f2,figname2);
saveas(f3,figname3);
