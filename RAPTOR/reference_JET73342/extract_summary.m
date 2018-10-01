%% Script snippit to pull interesting data from out0 RAPTOR output structure. Put this near RAPTOR_predictive/RAPTOR_out
% Initial kinetic profiles
summary.rho = out0.rho;
summary.te0 = out0.te(:, 1);
summary.ti0 = out0.ti(:, 1);
summary.ne0 = out0.ne(:, 1);

% Actuators (constant for all times)
summary.Ip = out0.Ip(:, 1);

% Boundary condition (constant for all times)
summary.teBC_rho = model.te.BC.rhoValue;
summary.teBC = v(model.te.BC.vind_value, 1);
summary.tiBC_rho = model.ti.BC.rhoValue;
summary.tiBC = v(model.ti.BC.vind_value, 1);
summary.neBC_rho = model.ne.BC.rhoValue;
summary.neBC = v(model.ne.BC.vind_value, 1);

% Sources (constant for all times)
summary.sne = out0.sne(:, 1);
summary.pnbe = out0.pnbe(:,1);
summary.pnbi = out0.pnbi(:,1);
summary.jnb  = out0.jnb(:,1);

% Final kinetic profiles
summary.te = out0.te(:, end);
summary.ti = out0.ti(:, end);
summary.ne = out0.ne(:, end);

summary.tgrid = tgrid;
