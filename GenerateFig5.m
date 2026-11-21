% Figure 5: one moving and one static source, array with element positions lambda/2 and lambda
% (the paper's geometry). Only estimated quantities are used (no true signals, DOAs or AR coefficients).
K = 2; N = 30; N_Int = 1000; Lks = [2,2]; lambda = 1;
D = lambda * [1/2; 1];          % element positions; spacing lambda is resolved by the lambda/2 pair
iSNR = 30;                    % [dB]
a = {[0.872 -0.550],[1.096 -0.870]};
cfg = Defaults();
cfg.L = Lks; cfg.a = a; cfg.N = N; cfg.NI = N_Int; cfg.d = D/lambda; cfg.iSNR = iSNR;
cfg.seed = 40; cfg.ffDOA = 0.99; cfg.thInit = [20 -20];
cfg.theta = [linspace(20,16,400) linspace(16,5.2,40) linspace(5.2,2,560); -20*ones(1,1000)];
res = JointKF_Corrected(cfg);
fprintf('DOA RMSE [deg] = %s,  AR RMSE = %s\n', mat2str(res.rmseDOA',3), mat2str(res.rmseAR',3));
f5 = figure;
plot(res.thEst(1,:),'r--'); hold on; plot(cfg.theta(1,:),'b-');
plot(res.thEst(2,:),'r--'); plot(cfg.theta(2,:),'b-');
ylabel('Source DOA [Degrees]'); xlabel('Time index (n)'); legend('Estimated','True');
saveas(f5,'Fig5.png');
