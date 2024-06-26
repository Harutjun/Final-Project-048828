[audioIn1,fs1] = audioread('Counting-16-44p1-mono-15secs.wav');

[audioIn2,fs2] = audioread('Rainbow-16-8-mono-114secs.wav');




audioIn1_clean = bandpass(audioIn1,[50,3000],fs1);
audioIn1_ds_clean =  downsample(audioIn1_clean,floor(fs1/fs2));

speechIdx = detectSpeech(audioIn1_ds_clean,fs2);
counting_up = [];
counting_down = [];
speechIdx(6,2) = 84500; % Fix the number six detection
for i = 1:10
    counting_up = cat(1,counting_up,audioIn1_ds_clean(speechIdx(i,1):speechIdx(i,2)));
    counting_down = cat(1,counting_down,audioIn1_ds_clean(speechIdx(11-i,1):speechIdx(11-i,2)));
end

% 
% player = audioplayer(x_post(1,:),fs2);
% play(player);
% detectSpeech(audioIn1_ds_clean,fs1);




N_Tot = numel(counting_down);
thetas = [linspace(40,20,floor(3*N_Tot/5)) linspace(20,7.5,floor(N_Tot/5)) linspace(7.5,0,N_Tot-floor(3*N_Tot/5)-floor(N_Tot/5))]; %Deg
thetas = cat(1,thetas,-20*ones(size(thetas)));

K = 2; % Num of sources
N = 30; % Num of Snapshoots per Interval
Lks = [2,2]; % order of sources AR processes
M = 2; % size of the subarrays
lambda = 1; % Wavelength in samples
D = (lambda/4) * ones(M,1); % Distances between elements in subarrays
F = zeros(sum(Lks));
Gamma_mat = zeros(K,sum(Lks));
CumSumLKs = cumsum(Lks);
DefineUtilityFunctions;
Xk = cat(2,counting_up,counting_down)';
iSNR = 30;
amp = sqrt(iSNR2ProcessNoiseVar(iSNR,1,from_dB(pow_dB(counting_up))));
Xk  = amp*Xk;


Xk_0 = Xk(:,1:N);
[arcoefs,E,Ks] = aryule(Xk_0(1,:),15);
pacf = -Ks;

f1 = figure;
stem(pacf,'filled')
xlabel('Lag')
ylabel('Partial ACF')
title('Partial Autocorrelation Sequence')
xlim([1 15])


conf = sqrt(2)*erfinv(0.95)/sqrt(N);
hold on
plot(xlim,[1 1]'*[-conf conf],'r')
hold off
grid
saveas(f1,sprintf('PACF_Speech_iSNR%2.f.png',iSNR));

a1 = aryule(Xk_0(1,:),Lks(1));
a2 = aryule(Xk_0(2,:),Lks(2));
a = {a1(2:end),a2(2:end)};

%% Test KF
alphas = get_alphas(deg2rad(thetas));

%%% Create the dynamics AR coefficient  matrix F
for k = 1:K
    Gamma_mat(k,CumSumLKs(k) - Lks(k) + 1) = 1;
end

CumSumLKs = cumsum(Lks);

for k = 1:K
    if k ==1
        F(k:CumSumLKs(k),k:CumSumLKs(k)) = [a{k}' [eye(Lks(k)-1) zeros(Lks(k)-1,1)]']';
    else
         F(CumSumLKs(k-1)+1:CumSumLKs(k),CumSumLKs(k-1)+1:CumSumLKs(k)) = [a{k}' [eye(Lks(k)-1) zeros(Lks(k)-1,1)]']';
    end
end

%%% Initialize Kalman Filter
x_post = [];
a_est1 = [];
a_est2 = [];
thetas_est = [];
P_prior =  10*eye(sum(Lks));
x_prior = zeros(sum(Lks),1);
initial_noise_cov = diag([amp^2 *ones(1,Lks(1)),amp^2* ones(1,Lks(2))]);
e_t = (1/sqrt(2))*(randn(2*M + 1,N_Tot) + 1j*randn(2*M + 1,N_Tot)); % Measurments noise (Complex White Gaussian)
err = [];
T_2Mp1 = 1/sqrt(2) * [eye(M) zeros(M,1) 1j*eye(M) ;...
                      zeros(1,M) 2 zeros(1,M);...
                      flip(eye(M)) zeros(M,1) -1j*flip(eye(M))];   


[init_alphas,d] = meshgrid(get_alphas(deg2rad([40,-20])),flip(D));
R1_DOA = 1e-1 * eye(K);
R2_DOA = 1e-1 * eye(K);
measure_cov = 0.5*eye(2*M+1) + 1j*0.5*eye(2*M+1);
for winIdx = 1:floor(N_Tot/N)

    startidx = 1+(winIdx*(N-1));
    stopidx = startidx + N - 1;
    Xk_n = Xk(:,startidx:stopidx);
    e_t_n = e_t(:,startidx:stopidx);

    X = [];
    Y= [];
    A = GetManifoldMatrix(alphas(:,startidx),D);
    for j=Lks(1):N
        xj = reshape(Xk_n(:,j-Lks(1)+1:j)',1,[])';
        yj = A*Gamma_mat*xj + e_t_n(:,j);
        Y = cat(2,Y,yj);
        Inov = calc_Inovation(x_prior, A*Gamma_mat,yj);
        Inov_Cov = calc_Inov_Cov(A*Gamma_mat,P_prior,measure_cov);
        K_Gain = update_KalmanGain(P_prior, A*Gamma_mat, Inov_Cov);
        x_post_j = calc_x_post(x_prior, K_Gain,Inov);
        P_post = calc_P_post(K_Gain,A*Gamma_mat,P_prior);
        err_j = calc_err(yj,A*Gamma_mat,x_post_j')';
        err = cat(1,err,sqrt(err_j*inv(eye(2*M+1))*err_j'));
        x_prior = calc_x_prior(x_post_j,F);
        P_prior = calc_P_prior(P_post,initial_noise_cov,F);

        if(~isempty(X))
            X(1,end-Lks(1)+1:end-1) = real(x_post_j(2:Lks(1)));
            X(2,end-Lks(2)+1:end-1) = real(x_post_j(Lks(1)+2:end));
            X = cat(2,X,real(x_post_j([1,Lks(1)+1])));
        else
            X = real(fliplr(reshape(x_post_j,Lks(2),[])'));
        end
    end

    x_post = cat(2,x_post,X);

    R1 = 1e-3 * eye(Lks(1));
    U1 = zeros(Lks(1),1);
    R2 = 1e-3 * eye(Lks(2));
    U2 = zeros(Lks(2),1);
    for j = Lks(1):N-Lks(1)-1

        [aj_est1,R1,U1,~,~] = QRD_RLS_Iteration(R1,U1,X(1,j-Lks(1) + 1 : j),X(1,j+1),0.98);
        [aj_est2,R2,U2,~,~] = QRD_RLS_Iteration(R2,U2,X(2,j-Lks(2) + 1 : j),X(2,j+1),0.98);
    end

    a_est1 = cat(2,a_est1,aj_est1);
    a_est2 = cat(2,a_est2,aj_est2);

    coeffGourpM_init1 = sqrt(2)*cos(d.*init_alphas);
    coeffGourpM_init2 = flipud(sqrt(2)*sin(d.*init_alphas));
    
    
    Middle_idx = M + 1;
    forgetting_factor = 0.98;
    Y_T = T_2Mp1' * Y;

    est_alpha = zeros(K,1);
    for m=1:M
        
        
        U1 = R1_DOA*coeffGourpM_init1(m,:)' ;
        U2 = R2_DOA*coeffGourpM_init2(m,:)' ;
    

        for j=1:numel(Y(1,:))
            GammaX = X(:,Lks(1)+j-1);
            [A_Tm_est1,R1_DOA,U1,~,~] = QRD_RLS_Iteration(R1_DOA,U1,GammaX',Y_T(Middle_idx-m,j),forgetting_factor);
    
            [A_Tm_est2,R2_DOA,U2,~,~] = QRD_RLS_Iteration(R2_DOA,U2,GammaX',Y_T(Middle_idx+m,j),forgetting_factor);
    
        end
        est_alpha = est_alpha + (atan(real(A_Tm_est2)./real(A_Tm_est1)))./D(m);
        
        
    end
    est_alpha = est_alpha/M;
    thetas_est = cat(1,thetas_est,get_thetas(est_alpha'));
    [init_alphas,d] = meshgrid(est_alpha',flip(D));
    coeffGourpM_init1 = sqrt(2)*cos(d.*init_alphas);
    coeffGourpM_init2 = flipud(sqrt(2)*sin(d.*init_alphas));
   
end




thetas_est = rad2deg(thetas_est);
err = real(err);

f2=figure;
plot(thetas(1,:),'b-');
hold on;
plot(thetas(2,:),'b-');
plot(N*(1:size(thetas_est,1)),thetas_est(:,1),'r--');
plot(N*(1:size(thetas_est,1)),thetas_est(:,2),'g--');
xlabel('sample');
ylabel('DOA [Deg]');
legend('Source 1','Source 2', 'Estimated Source 1', 'Estimated Source 2');
% saveas(f2,sprintf('DOA_Speech_iSNR%2.f.png',iSNR));