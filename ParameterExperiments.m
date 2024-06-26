%% iSNR
overrideParams = 1;
K = 2; % Num of sources
N = 30; % Num of Snapshoots per Interval
N_Int = 200; % Num of Intervarls
Lks = [2,2]; % order of sources AR processes
M = 1; % size of the subarrays
lambda = 1; % Wavelength
D = (lambda/4) * ones(M,1); % Distances between elements in subarrays
iSNR = 1; % [dB] 10*log10(Power_Signal/Power_noise)
a1 = [0.872 -0.550]; % Values from the paper
a2 = [1.096 -0.870]; % Values from the paper
a = {a1,a2};

figname2 = sprintf('Fig2_iSNR%2.f.png',iSNR);
figname3 = sprintf('Fig3_iSNR%2.f.png',iSNR);
GenerateFigs2And3;


%% M
overrideParams = 1;
K = 2; % Num of sources
N = 30; % Num of Snapshoots per Interval
N_Int = 200; % Num of Intervarls
Lks = [2,2]; % order of sources AR processes
M = 4; % size of the subarrays
lambda = 1; % Wavelength
D = (lambda/4) * ones(M,1); % Distances between elements in subarrays
iSNR = 30; % [dB] 10*log10(Power_Signal/Power_noise)
a1 = [0.872 -0.550]; % Values from the paper
a2 = [1.096 -0.870]; % Values from the paper
a = {a1,a2};

figname2 = sprintf('Fig2_M%2.f.png',M);
figname3 = sprintf('Fig3_M%2.f.png',M);
GenerateFigs2And3;


%% AR Parameters
overrideParams = 1;
K = 2; % Num of sources
N = 30; % Num of Snapshoots per Interval
N_Int = 200; % Num of Intervarls
Lks = [3,3]; % order of sources AR processes
M = 2; % size of the subarrays
lambda = 1; % Wavelength
D = (lambda/4) * ones(M,1); % Distances between elements in subarrays
iSNR = 30; % [dB] 10*log10(Power_Signal/Power_noise)
a1 = [0.872 -0.550 0.223]; % Values from the paper
a2 = [1.096 -0.870 0.315]; % Values from the paper
a = {a1,a2};

figname2 = sprintf('Fig2_AR%1.f.png',numel(a1));
figname3 = sprintf('Fig3_AR%1.f.png',numel(a1));
GenerateFigs2And3;