K = 2; % Num of sources
N = 3; % Num of Snapshoots per Interval
N_Int = 1000; % Num of Intervarls
Lks = [2,2]; % order of sources AR processes
M = 2; % size of the subarrays
lambda = 1; % Wavelength
D = (lambda/2) * ones(M,1); % Distances between elements in subarrays
iSNR = 10; % [dB] 10*log10(Power_Signal/Power_noise)
a1 = [0.872 -0.550]; % Values from the paper
a2 = [1.096 -0.870]; % Values from the paper
a = {a1,a2};
DefineUtilityFunctions;
rng(40); %For Reproducability
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


N_Tot = N*N_Int;
thetas = [linspace(10,9,400) linspace(9,7.5,40) linspace(7.5,6.1,560)]; %Deg

thetas = cat(1,thetas,zeros(size(thetas)));


Vars_k = cellfun(@(a) iSNR2ProcessNoiseVar(iSNR,1,mean(filter(1,[1 -a] ,randn(1,1e8)).^2)),{a1,a2});

Xk = [];
v_t = randn(K,N_Tot + 1); % Process Noise 

for k = 1:K
    Xk = cat(1,Xk,filter(sqrt(Vars_k(k)),[1, -a{k}],v_t(k,:)));
end


e_t = (1/sqrt(2))*(randn(2*M + 1,N_Tot) + 1j*randn(2*M + 1,N_Tot));
Y = [];
X = [];

A = GetManifoldMatrix(get_alphas(deg2rad(thetas(:,1)))',D);
for i=K:N_Tot
    xj = reshape(fliplr(Xk(:,i-K+1:i))',1,[])';
    if(~mod(i,N))
        A = GetManifoldMatrix(get_alphas(deg2rad(thetas(:,floor(i/N))))',D);
    end
    yj = A*Gamma_mat*xj + e_t(:,i-1);
    X = cat(2,X,xj);   
    Y = cat(2,Y,yj);
end


[init_alphas,d] = meshgrid(get_alphas(deg2rad([10,0])),flip(D));
coeffGourpM_init1 = sqrt(2)*cos(d.*init_alphas);
coeffGourpM_init2 = flipud(sqrt(2)*sin(d.*init_alphas));

thetas_est = [];
Middle_idx = M + 1;
forgetting_factor =0.99;

T_2Mp1 = 1/sqrt(2) * [eye(M) zeros(M,1) 1j*eye(M) ;...
                  zeros(1,M) 2 zeros(1,M);...
                  flip(eye(M)) zeros(M,1) -1j*flip(eye(M))];   
Y_T = T_2Mp1' * Y;
GammaX = Gamma_mat*X;
R1 = 1e1 * eye(K);
R2 = 1e1 * eye(K);
for n=1:N:N_Tot-N
    est_alpha = zeros(K,1);

    for m=1:M
        U1 = R1*coeffGourpM_init1(m,:)' ;
        U2 = R2*coeffGourpM_init2(m,:)' ;
   
        for j=1:N
    
            [A_Tm_est1,R1,U1,~,~] = QRD_RLS_Iteration(R1,U1,GammaX(:,n+j-1)',Y_T(Middle_idx-m,n+j-1),forgetting_factor);
    
            [A_Tm_est2,R2,U2,~,~] = QRD_RLS_Iteration(R2,U2,GammaX(:,n+j-1)',Y_T(Middle_idx+m,n+j-1),forgetting_factor);
    
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
f4 = figure;
plot(1:numel(thetas_est(:,1)),thetas_est(:,1),'r--');
hold on;
plot(1:N_Int,thetas(1,:),'b-')
plot(1:numel(thetas_est(:,1)),thetas_est(:,2),'r--');
plot(1:N_Int,thetas(2,:),'b-')
ylabel('Source DOA [Degrees]');
xlabel('Time index (n)');
legend('Estimated','True');
saveas(f4,'Fig4.png');
