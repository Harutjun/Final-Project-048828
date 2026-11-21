% Figures 2 and 3: static sources at 0 and 20 deg, AR coefficient and DOA estimates over intervals.
% The estimator (JointKF_Corrected) uses only estimated quantities: the KF runs with A(theta_hat) and
% F(a_hat) from the previous interval, and both regressions use the KF's state estimates.
if ~exist('overrideParams','var')
    K = 2; N = 30; N_Int = 200; Lks = [2,2];
    M = 1; lambda = 1;
    D = (lambda/4) * ones(M,1);     % element positions (one side of the array)
    iSNR = 30;                      % [dB]
    a1 = [0.872 -0.550]; a2 = [1.096 -0.870]; a = {a1,a2};
    aInit = {[0.772 -0.45],[0.96 -0.77]};   % initial AR guess (paper)
    figname2 = 'Fig2.png'; figname3 = 'Fig3.png';
end
cfg = Defaults();
cfg.L = Lks; cfg.a = a; cfg.aInit = aInit; cfg.N = N; cfg.NI = N_Int;
cfg.d = D/lambda; cfg.iSNR = iSNR; cfg.seed = 20;
cfg.theta = repmat([0;20],1,N_Int); cfg.thInit = [5 25];
res = JointKF_Corrected(cfg);
fprintf('DOA RMSE [deg] = %s,  AR RMSE = %s\n', mat2str(res.rmseDOA',3), mat2str(res.rmseAR',3));

f2 = figure; sgtitle('Figure 2');
for k = 1:2
    subplot(1,2,k); hold on;
    rows = sum(Lks(1:k-1)) + (1:Lks(k));
    for r = rows, histogram(res.aEst(r,:),20); end
    plot(a{k},0,'*r','MarkerSize',10);
    ylabel('Histogram'); title(sprintf('AR coefficients of source %d',k));
    legend([repmat({'Estimated'},1,numel(rows)) {'True'}]);
end
f3 = figure; hold on;
for k = 1:K, histogram(res.thEst(k,:),20); end
plot(cfg.theta(:,1),zeros(K,1),'r*','MarkerSize',15);
sgtitle('Figure 3'); legend('Estimated','Estimated','True');
xlabel('DOA [Degree]'); ylabel('Histogram');
saveas(f2,figname2); saveas(f3,figname3);
