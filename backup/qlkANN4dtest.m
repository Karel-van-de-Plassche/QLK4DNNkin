% Test program for Artificial Neural Network QUALIKIZ transport model (4D input version)
%NOTE: this is for ITG only, thus no ETG fluxes


% Load neural networks 
load('kin_e_5D_ITG_ief'); parametre.netief=net; 
load('kin_e_5D_ITG_eef'); parametre.neteef=net;
load('kin_e_5D_ITG_dfe'); parametre.netdfe=net;
load('kin_e_5D_ITG_vte'); parametre.netvte=net;
load('kin_e_5D_ITG_vce'); parametre.netvce=net;

%whenzero: since NN was only trained on positive fluxes, they will extrapolate
%to negative flux (which is unphysical for heat fluxes and particle diffusion).
%thus, we zero out all fluxes when
% whenzero=1: ion heat fluxes are negative 
% whenzero=2: ion or electron heat fluxes are negative
% whenzero=3: ion or electron heat or electron particle diffusion is negative 
parametre.whenzero = 1; %default value

parametre.zerooutpinch = 0; %if 1, then zeros out positive pinches (not physical for ITG)

%fill in profile structure
scann = 24; %size of scan variable
prof.rlti=linspace(2,13,scann)';
prof.q=2.*ones(scann,1);
prof.s=1.*ones(scann,1);
prof.tite=1.*ones(scann,1);

% These are constants for the NN database
prof.te=8.*ones(scann,1); %in KeV
prof.ne=5.*ones(scann,1); %in 10^19 m^-3
prof.ni=5.*ones(scann,1); %in 10^19 m^-3
prof.rlte=6.*ones(scann,1);
prof.rlne=2.*ones(scann,1); 

%fill in scalar variable structure (constants in NN database)
scalar.Amain=2; %in amu
scalar.b0=3; %in T
scalar.r0=3; %in m
scalar.a=1; %in m

[qi_GB,qe_GB,pfe_GB] = qlkANN_driver(parametre,scalar,prof);
fsize=12;
figure;
subplot(1,2,1);
plot(prof.rlti,qi_GB,'r',prof.rlti,qe_GB,'b'); hold on
plot(prof.rlti,qi_GB,'r.',prof.rlti,qe_GB,'b.'); 
l1=legend('Q_i','Q_e','Location','NorthWest'); legend('boxoff');
t1=xlabel('R/L_{Ti}');
t2=ylabel('GB heat flux');
t3=title('q=2, s=1, Ti/Te=1');
set(gca,'FontSize',fsize);
set(t1,'FontSize',fsize);
set(t2,'FontSize',fsize);
set(t3,'FontSize',fsize);
set(l1,'FontSize',fsize);

subplot(1,2,2);
plot(prof.rlti,pfe_GB,'g'); hold on
plot(prof.rlti,pfe_GB,'g.');
l1=legend('\Gamma_e','Location','NorthWest'); legend('boxoff');
t1=xlabel('R/L_{Ti}');
t2=ylabel('GB electron particle flux');
t3=title('q=2, s=1, Ti/Te=1');
set(gca,'FontSize',fsize);
set(t1,'FontSize',fsize);
set(t2,'FontSize',fsize);
set(t3,'FontSize',fsize);
set(l1,'FontSize',fsize);
