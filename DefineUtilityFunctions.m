get_alphas = @(thetas) 2*pi * sin(thetas) / lambda;  % theta in Radians
get_thetas = @(alphas) asin(alphas* (lambda/(2*pi)));
%%% KF Update Functions
calc_Inovation = @(x_cur_prior, A, y) y - A * x_cur_prior;
calc_Inov_Cov = @(A,P_cur_prior, meas_noise_Cov) A * P_cur_prior * A' + meas_noise_Cov;
update_KalmanGain = @(P_cur_prior, A, Inov_Cov) P_cur_prior * A' * inv(Inov_Cov);
calc_x_post = @(x_cur_prior, kalmanGain, Innovation) x_cur_prior + kalmanGain * Innovation;
calc_P_post = @(KalmanGain,A,P_cur_prior) (eye(size(KalmanGain,1),size(A,2)) - KalmanGain * A)*P_cur_prior;
calc_err = @(y,A,x_post) y - A * x_post';
%%% KF Prediction Functions
calc_x_prior = @(x_prev_post,F) F * x_prev_post;
calc_P_prior = @(P_prev_post, process_noise_Cov, F) F*P_prev_post*F' +  process_noise_Cov;

%%% power calculation utilities and conversion to dB
pow = @(sig) mean(abs(sig.^2)); 
to_dB = @(x) 10*log10(x);
from_dB = @(x) 10.^(x/10);
pow_dB = @(sig) to_dB(pow(sig));
iSNR2ProcessNoiseVar = @(iSNR,Power_noise,Power_NormalNoiseProcess) (from_dB(iSNR))./Power_NormalNoiseProcess;


