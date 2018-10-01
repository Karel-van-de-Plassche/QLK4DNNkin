RAPTOR parameters for comparing with F.Felici et al. NF2018.

Evolved for 3 seconds with timesteps of 0.2s.

B0 = 2.7
R0 = 2.93
Zeff = 1.8 (from CRONOS)

## Model
- No sawteeth (saw)
- chie, de and ve from QLKANN-4Dkin
  - ve positive values clipped to 0
  - NN inputs not constrained
  - Minimum value added to computed transport, proxy for NC transport
    - chiimin = 0.2
    - chiemin = 0.15
    - vemin = NaN
    - demin = 0.1
  - NN output is smoothed in space

## Boundary conditions
- te, ti and ne fixed at pedestal top rho_ped=0.85 from CRONOS run:
  - te(rho=0.85) = 1.0243e+03
  - ti(rho=0.85) = 1.0401e+03
  - te(rho=0.85) = 8.5896e+19
- Transport at LCFS fixed, interpolated in pedestal
  - chie(rho=1) = 3
  - chii(rho=1) = 5
  - de(rho=1) = 0.15
  - ve(rho=1) = -4
  - Interpolation for values rho=[rho_ped, 1], for transport coefficient X:
    - lamped = 1-(rho-rho_ped).^2/(1-rho_ped).^2;
    - X = X * lamped + (1-lamped) * X(rho=1)

## Sources
- No ICRH
- No line-radiation profile (prad/prad_sink)
- Prescribed NBI power and currents (pnbe, pnbi, jnb)
- Electron particle source (sne_pr) ((where does it come from?))
- Constant current (Ip/U(1,:))

## Self-consistently calculated
- Brehmstrahlung (pbrem)
- Electron-Ion equipartition (pei)

## Equilibrium
- From CHEASE
