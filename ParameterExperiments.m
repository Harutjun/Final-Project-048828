%% iSNR
overrideParams = 1;
K = 2; N = 30; N_Int = 200; Lks = [2,2]; M = 1; lambda = 1;
D = (lambda/4) * ones(M,1);
iSNR = 1;
a = {[0.872 -0.550],[1.096 -0.870]}; aInit = {[0.772 -0.45],[0.96 -0.77]};
figname2 = sprintf('Fig2_iSNR%2.f.png',iSNR);
figname3 = sprintf('Fig3_iSNR%2.f.png',iSNR);
GenerateFigs2And3;

%% M  (distinct element positions lambda*(1:M)/(2M))
overrideParams = 1;
K = 2; N = 30; N_Int = 200; Lks = [2,2]; M = 4; lambda = 1;
D = lambda*(1:M)'/(2*M);
iSNR = 30;
a = {[0.872 -0.550],[1.096 -0.870]}; aInit = {[0.772 -0.45],[0.96 -0.77]};
figname2 = sprintf('Fig2_M%2.f.png',M);
figname3 = sprintf('Fig3_M%2.f.png',M);
GenerateFigs2And3;

%% AR Parameters (order 3)
overrideParams = 1;
K = 2; N = 30; N_Int = 200; Lks = [3,3]; M = 2; lambda = 1;
D = lambda*(1:M)'/(2*M);
iSNR = 30;
a1 = [0.872 -0.550 0.223]; a2 = [1.096 -0.870 0.315]; a = {a1,a2};
aInit = {[0.77 -0.45 0.2],[0.96 -0.77 0.3]};   % initial guess has the same length as the AR order
figname2 = sprintf('Fig2_AR%1.f.png',numel(a1));
figname3 = sprintf('Fig3_AR%1.f.png',numel(a1));
GenerateFigs2And3;
