function out = JointKF_Corrected(cfg)
% Joint KF + QRD-RLS estimation of source signals, AR coefficients and DOAs (Gu et al. style).
% No true quantity is used unless an oracle flag (oracle / orA / orF / orS) is set for ablation.
% Optional cfg fields: nIter (passes per window, default 1), ffDOA, ffAR, delta, p0, qScale.
K = 2; L = cfg.L; N = cfg.N; NI = cfg.NI; lambda = 1;
d = cfg.d(:); M = numel(d);
aTrue = cfg.a; aInit = cfg.aInit; thTrue = cfg.theta; thHat = cfg.thInit(:);
g = @(f,def) getf(cfg,f,def);
ffAR = g('ffAR',0.995); ffDOA = cfg.ffDOA; delta = g('delta',1e-2); nIter = g('nIter',1);
p0 = g('p0',10); qScale = g('qScale',1);
orA = cfg.oracle || g('orA',0); orF = cfg.oracle || g('orF',0); orS = cfg.oracle || g('orS',0);
rng(cfg.seed);
alph = @(th) 2*pi*sind(th)/lambda;
Amat = @(th) manifold(alph(th(:).'), d);
T = 1/sqrt(2)*[eye(M) zeros(M,1) 1j*eye(M); zeros(1,M) sqrt(2) zeros(1,M); flip(eye(M)) zeros(M,1) -1j*flip(eye(M))];

NT = N*NI; sig2 = zeros(1,K); s = zeros(K,NT);
for k=1:K
  h = impz(1,[1 -aTrue{k}],5000); sig2(k) = 10^(cfg.iSNR/10)/sum(abs(h).^2);
  v = sqrt(sig2(k)/2)*(randn(1,NT+200)+1j*randn(1,NT+200));
  tmp = filter(1,[1 -aTrue{k}],v); s(k,:) = tmp(201:end);
end
Y = zeros(2*M+1,NT);
for t=1:NT
  Y(:,t) = Amat(thTrue(:,ceil(t/N)))*s(:,t) + (randn(2*M+1,1)+1j*randn(2*M+1,1))/sqrt(2);
end
YT = T'*Y;

Ltot = sum(L); off = [0 cumsum(L)];
Gam = zeros(K,Ltot); Qd = zeros(Ltot);
for k=1:K, Gam(k,off(k)+1) = 1; Qd(off(k)+1,off(k)+1) = qScale*sig2(k); end
Fof = @(a) blkdiag(comp(a{1}), comp(a{2}));
Rm = eye(2*M+1);

aHat = aInit; xh = zeros(Ltot,1); P = p0*sum(sig2)*eye(Ltot);
arR = cell(1,K); arU = cell(1,K);
for k=1:K, arR{k} = delta*eye(L(k)); arU{k} = arR{k}*aInit{k}(:); end
dR = cell(2,M); dU = cell(2,M); a0 = alph(thHat.');
for m=1:M
  dR{1,m} = delta*eye(K); dU{1,m} = dR{1,m}*(sqrt(2)*cos(d(m)*a0)).';
  dR{2,m} = delta*eye(K); dU{2,m} = dR{2,m}*(sqrt(2)*sin(d(m)*a0)).';
end
thEst = zeros(K,NI); aEst = zeros(sum(L),NI); sHat = zeros(K,NT);

for n=1:NI
  idx = (n-1)*N+(1:N);
  if orF, F = Fof(aTrue); else, F = Fof(aHat); end
  xh0 = xh; P0 = P; dR0 = dR; dU0 = dU; thIt = thHat;
  for it=1:nIter
    if orA, Ause = Amat(thTrue(:,n)); else, Ause = Amat(thIt); end
    H = Ause*Gam; xh = xh0; P = P0;
    for t=idx
      xp = F*xh; Pp = F*P*F' + Qd; S = H*Pp*H' + Rm; Kg = Pp*H'/S;
      xh = xp + Kg*(Y(:,t)-H*xp); P = (eye(Ltot)-Kg*H)*Pp; P = (P+P')/2;
      sHat(:,t) = Gam*xh;
    end
    if orS, sReg = s; else, sReg = sHat; end
    dR = dR0; dU = dU0; phi = zeros(K,M);
    for m=1:M
      rc = M+1-m; rs = 2*M+2-m;
      for t=idx
        [dR{1,m},dU{1,m}] = qrdrls(dR{1,m},dU{1,m},sReg(:,t).',YT(rc,t),ffDOA);
        [dR{2,m},dU{2,m}] = qrdrls(dR{2,m},dU{2,m},sReg(:,t).',YT(rs,t),ffDOA);
      end
      phi(:,m) = atan2(real(dR{2,m}\dU{2,m}), real(dR{1,m}\dU{1,m}));   % d_m*alpha mod 2pi
    end
    alphaHat = resolveAlpha(phi,d,g('resolve',1));
    thIt = asind(max(min(alphaHat*lambda/(2*pi),1),-1));
  end
  thHat = thIt; thEst(:,n) = thHat;
  for k=1:K   % AR update from the final-pass signal estimates
    for t=idx
      if t<=L(k), continue; end
      past = sReg(k,t-1:-1:t-L(k));
      [arR{k},arU{k}] = qrdrls(arR{k},arU{k},real(past),real(sReg(k,t)),ffAR);
      [arR{k},arU{k}] = qrdrls(arR{k},arU{k},imag(past),imag(sReg(k,t)),ffAR);
    end
    aHat{k} = real(arR{k}\arU{k}).';
  end
  aEst(:,n) = [aHat{:}].';
end
out.thEst = thEst; out.aEst = aEst; out.thTrue = thTrue;
burn = ceil(NI/5);
out.rmseDOA = sqrt(mean((thEst(:,burn:end)-thTrue(:,burn:end)).^2,2));
out.rmseAR = sqrt(mean((aEst(:,burn:end)-[aTrue{:}].').^2,2));
end
function alpha = resolveAlpha(phi,d,doResolve)
% phi(:,m) = d_m*alpha mod 2pi. The smallest spacing (<= lambda/2) is unambiguous and fixes the
% integer ambiguity of the larger spacings; all pairs are then averaged (the paper's redundancy).
[K,M] = size(phi); alphas = zeros(K,M);
[~,m0] = min(d); a0 = phi(:,m0)/d(m0);
for m=1:M
  if doResolve, kk = round((d(m)*a0 - phi(:,m))/(2*pi)); else, kk = 0; end
  alphas(:,m) = (phi(:,m)+2*pi*kk)/d(m);
end
alpha = mean(alphas,2);
end
function v = getf(c,f,def), if isfield(c,f), v = c.(f); else, v = def; end, end
function [R,U] = qrdrls(R,U,x,y,lam)
K = numel(x); [~,Rb] = qr([sqrt(lam)*R sqrt(lam)*U; x y]); R = Rb(1:K,1:K); U = Rb(1:K,K+1);
end
function Fk = comp(a), p = numel(a); Fk = [a(:).'; eye(p-1) zeros(p-1,1)]; end
function A = manifold(alphas,d)
A1 = exp(1j*flip(d)*alphas); A = [A1; ones(1,numel(alphas)); conj(flipud(A1))];
end
