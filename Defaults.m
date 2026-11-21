function cfg = Defaults()
% Defaults from tune.m / tune2.m (seed 1): 25 KF+DOA passes per window, forgetting 0.995.
cfg.L=[2 2]; cfg.a={[0.872 -0.550],[1.096 -0.870]}; cfg.aInit={[0.772 -0.45],[0.96 -0.77]};
cfg.oracle=false; cfg.seed=20; cfg.ffDOA=0.995; cfg.nIter=25; cfg.p0=10; cfg.delta=1e-2;
end
