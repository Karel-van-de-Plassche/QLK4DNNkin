% Artificial Neural Network QUALIKIZ transport model. 4D kinetic electron version
% Juan Redondo and Jonathan Citrin, Aug 2015

function [qi_GB,qe_GB,pfe_GB] = qlkANN_driver(parametre,scalar,prof)

%%%%%%%%%%%%%%%%%%%%%%%%% Call Qualikiz ANN (accepts columns)%%%%%%%%%%%%%%%%%%

[chii] = qlkANNk(prof.rlti,prof.q,prof.tite,prof.s,parametre.netief.max_qlk,parametre.netief.min_qlk,parametre.netief,parametre.netief.prepros);
[chie] = qlkANNk(prof.rlti,prof.q,prof.tite,prof.s,parametre.neteef.max_qlk,parametre.neteef.min_qlk,parametre.neteef,parametre.neteef.prepros);
[D] = qlkANNk(prof.rlti,prof.q,prof.tite,prof.s,parametre.netdfe.max_qlk,parametre.netdfe.min_qlk,parametre.netdfe,parametre.netdfe.prepros);
[Vt] = qlkANNk(prof.rlti,prof.q,prof.tite,prof.s,parametre.netvte.max_qlk,parametre.netvte.min_qlk,parametre.netvte,parametre.netvte.prepros);
[Vc] = qlkANNk(prof.rlti,prof.q,prof.tite,prof.s,parametre.netvce.max_qlk,parametre.netvce.min_qlk,parametre.netvce,parametre.netvce.prepros);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

qel=1.6e-19;
chifac = (prof.te.*1e3.*qel).^1.5.*sqrt(scalar.Amain.*1.67e-27)./(qel.^2.*scalar.b0.^2.*scalar.a);

% This kinetic electron 4D NN was (mistakingly) fit to unnormalized D and V. We must normalize back to GB using the
% Te, B0, and a values used when constructing the NN database, and then unnormalize again using
% the simulation values

% NN training set values: 
%Te = 8 keV , a = 1 m, B0 = 3 T, Amain = 2, n = 5e19 [m^-3], R/Ln=2, R/LTe=6, x=0.5, Zeff=1
chifacNN = (8e3.*qel).^1.5.*sqrt(2.*1.67e-27)./(qel.^2.*3.^2.*1);

chii=(chifac.*chii)'; %convert to SI
chie=(chifac.*chie)'; %convert to SI
D=D .* chifac./chifacNN; %convert from SI to GB with NN units, then to SI with simulation units
V=(Vt+ Vc) .*chifac./chifacNN .* 3./scalar.r0;
if parametre.zerooutpinch == 1 %zero out outward pinch (not expected for ITG)
  V(V>0)=0;
end
if parametre.whenzero == 1
  filter = find(chii<0);
elseif parametre.whenzero == 2
  filter = find(chii<0 | chie<0);
elseif parametre.whenzero == 3
  filter = find(chii<0 | chie<0 | D<0);
end
chii(filter)=0;
chie(filter)=0;
D(filter)=0;
V(filter)=0;

dndr = -prof.rlne./scalar.r0.*(prof.ne*1e19);
dtidr = -prof.rlti./scalar.r0.*(prof.te.*prof.tite*1e3*qel);
dtedr = -prof.rlte./scalar.r0.*(prof.te.*1e3*qel);

qe_GB = -chie'./chifac.*dtedr./(prof.te*1e3*qel/scalar.r0);
qi_GB = -chii'./chifac.*dtidr./(prof.te.*prof.tite.*1e3*qel/scalar.r0);
pfe_GB = (-D.*dndr+V.*prof.ne*1e19)./chifac./(prof.ne*1e19./scalar.r0);

%[prof.rlti qi_GB qe_GB pfe_GB]

