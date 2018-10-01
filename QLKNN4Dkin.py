import os
from itertools import chain
from warnings import warn
import json

import numpy as np
import scipy as sc
import scipy.io as sio
import matplotlib.pyplot as plt
from IPython import embed
import pandas as pd

def sigmo(x):
    #nonlinear neuron function. 
    return 2./(1 + np.exp(-2 * x)) - 1

def NNoutput(Ati, q, Ti_Te, smag, net, Ati_shift=0):
    max_qlk = net['max_qlk']
    min_qlk = net['min_qlk']
    prepros = net['prepros']
    #initialize input matrix in order expected by the trained network. 
    invec = np.vstack([q, smag, Ti_Te, Ati])

    #find size of input vector (if Ati, q, Ti_Te, shear are not same length, the function will crash)
    npoints = len(Ati)

    #Normalize input matrix to match NN input normalization
    scalefac = QLKNN4Dkin.scalefac
    dispfac  = QLKNN4Dkin.dispfac

    #normalize input vector to match input normalization in NN
    unorm = np.multiply(np.add(invec, -dispfac), 1./scalefac)

    # Shift R/LTi thgresholds for threshold matching
    b1 = net['b1'] + Ati_shift * net['IW'][:,[-1]] / scalefac[-1]

    #calculate regression neural network piece by piece, for all radii
    g = np.add(np.dot(net['IW'], unorm), b1)
    sg = sigmo(g)
    f = np.add(np.dot(net['L1W'], sg), net['b2'])
    sf = sigmo(f)
    output = np.add(np.dot(net['L2W'], sf), net['b3'])

    # if outputs processing between 0 and 1
    if prepros == 0:
        NNout = output * (max_qlk - min_qlk) + min_qlk
    # if outputs processing between -1 a +1
    elif prepros == -1:
        NNout = np.multiply((output + 1), (max_qlk - min_qlk))/2 + min_qlk
    return NNout

def net_to_dict(net):
    netdict = {}
    for name in ['IW', 'L1W', 'L2W', 'b1', 'b2', 'b3']:
        netdict[name] = net['net'][name][0][0]
    for name in ['max_qlk', 'min_qlk', 'prepros']:
        netdict[name] = net['net'][name][0][0][0][0]
    return netdict

class QLKNN4Dkin():
    validity_range = {'Zeffx':  (1, 1),
                      'rln':     (2, 2), # == An
                      'rlte':    (6, 6), # == Ate
                      'rlti':    (2, 12), # ==Ati
                      'Nustar': (0.0084058, 0.0084058),
                      'tite':  (.3, 3), # ==Ti_Te
                      'q':     (1, 5),
                      'smag':   (.1, 3),
                      'x':      (.5, .5)}
    scalefac = np.array([2 , 1.45 , 1.35 , 5], ndmin=2).T
    dispfac  = np.array([3 , 1.55 , 1.65 , 7], ndmin=2).T
    feature_names = ['q', 'smag', 'Ti_Te', 'Ati']

    def __init__(self, whenzero=4, zerooutpinch=1):
        """
        whenzero: since NN was only trained on positive fluxes, they will extrapolate
        to negative flux (which is unphysical for heat fluxes and particle diffusion).
        thus, we zero out all fluxes when:
            - whenzero=1: ion heat fluxes are negative
            - whenzero=2: ion or electron heat fluxes are negative
            - whenzero=3: ion or electron heat or electron particle diffusion is negative 
            - whenzero=4: electron heat fluxes are negative

        zerooutpinch: if 1, then zeros out positive pinches (not physical for ITG)
        """
        self.whenzero = whenzero
        self.zerooutpinch = zerooutpinch
        netnames = ['ief', 'eef', 'dfe', 'vte', 'vce']
        matdict = sio.loadmat('./RAPTOR/implementation/qlkANN/kinetic_e_5D_ITG.mat')
        self.nets = {}
        netarray = matdict['net'][0,0]
        for ii, (fieldname, type) in enumerate(netarray.dtype.descr):
            split = fieldname.split('_')
            if len(split) >= 2:
                var = '_'.join(split[:-1])
                netname = 'net' + split[-1]
                if netname not in self.nets:
                    self.nets[netname] = {}
                self.nets[netname][var] = netarray[ii]

    def get_fluxes(self, te, ti, rlte, rlti, q, smag, R0, a0, B0, Amain, output_SI=True, clip_input=None, clip_input_margin=None, shift_threshold=None,  **kwargs):
        """
        Args:
            te:    Electron temperature [eV]
            ti:    Ion temperature [eV]
            rlte:  Normalized electron temperature gradient: -(R0 / Te) * (dTe/dr)
            rlti:  Normalized ion temperature gradient: -(R0 / Ti) * (dTi/dr)
            q:     Safety factor
            smag:  Magnetic shear: r * (q' / q)
            R0:    Midplane-averaged major radius of last-closed-flux-surface [m]
            a0:    Midplane-averaged minor radius of last-closed-flux-surface [m]
            B0:    Magnetic field at magnetic axis [T]
            Amain: Mass of main ion species (in proton masses)

        Kwargs:
            output_SI:  Output transport coefficients in SI units
            clip_input: Clip NN input to training values

        Returns:
            chie: Electron heat conductivity in GB or SI (see `output_SI`)
            chii: Ion heat conductivity in GB or SI (see `output_SI`)
            de:   Electron particle diffusivity in GB or SI (see `output_SI`)
            ve:   Electron total particle pinch in GB or SI (see `output_SI`)
        """
        # Temperatures and temperature gradients in SI
        qel = 1.6e-19;
        mp = 1.67e-27
        teSI  = te*qel
        tiSI  = ti*qel

        #Define gyroBohm normalization. NN outputs are in GB normalized units
        #chiGB = sqrt(mi)/(a*e^2*B^2)*Te^1.5
        chifac = np.sqrt(Amain*mp)/(qel**2*B0**2*a0)

        # Particle transport coefficients were (accidentally) trained in SI units
        # We must revert back to GB units using the dimensional units employed in the training,
        #and then revert to SI using the dimensional units in our current simulation
        # NN training set values: Te = 8 keV , a = 1 m, B0 = 3 T, Amain = 2, n = 5e19 [m^-3], R/Ln=2, R/LTe=6, x=0.5, Zeff=1
        chiGBNN = (8e3 * qel)**1.5 * np.sqrt(2 * mp) / (qel**2 * 3**2 * 1)

        #rlti = -R0 * tiSIp / tiSI
        tite = tiSI / teSI

        clip_input_base = {
            'tite': True,
            'rlti': True,
            'q':    True,
            'smag': True,
        }
        if clip_input is None:
            clip_input = clip_input_base
        else:
            clip_input = clip_input_base.update(clip_input)

        clip_input_margin_base = {
            'tite': .95,
            'rlti': .95,
            'q':    .95,
            'smag': .95,
        }
        if clip_input_margin is None:
            clip_input_margin = clip_input_margin_base
        else:
            clip_input_margin = clip_input_margin_base.update(clip_input_margin)

        inp = {'rlti': rlti,
               'q': q,
               'tite': tite,
               'smag': smag
               }
        for name, value in inp.items():
            if clip_input[name]:
                inp_min = self.validity_range[name][0] + (1-clip_input_margin[name]) * np.abs(self.validity_range[name][0])
                inp_max = self.validity_range[name][1] - (1-clip_input_margin[name]) * np.abs(self.validity_range[name][1])
                inp[name][value < inp_min] = inp_min
                inp[name][value > inp_max] = inp_max

        shift_threshold_base = {'chii': 0,
                                'chie': 0,
                                'DV': 0}
        if shift_threshold is None:
            shift_threshold = shift_threshold_base
        else:
            shift_threshold = shift_threshold_base.update(shift_threshold)

        chiiGB = NNoutput(inp['rlti'], inp['q'], inp['tite'], inp['smag'], self.nets['netief'], Ati_shift=shift_threshold['chii'])
        chieGB = NNoutput(inp['rlti'], inp['q'], inp['tite'], inp['smag'], self.nets['neteef'], Ati_shift=shift_threshold['chie'])
        deSI   = NNoutput(inp['rlti'], inp['q'], inp['tite'], inp['smag'], self.nets['netdfe'], Ati_shift=shift_threshold['DV'])
        vteSI  = NNoutput(inp['rlti'], inp['q'], inp['tite'], inp['smag'], self.nets['netvte'], Ati_shift=shift_threshold['DV'])
        vceSI  = NNoutput(inp['rlti'], inp['q'], inp['tite'], inp['smag'], self.nets['netvce'], Ati_shift=shift_threshold['DV'])
        # convert from SI to GB with dimensional units used in training
        deGB   = deSI / chiGBNN
        vteGB  = vteSI / chiGBNN * 3
        vceGB  = vceSI / chiGBNN * 3
        veGB = vteGB + vceGB

        #zeff = 1.
        #c1 = (6.9224e-5 * zeff * ne * qx * Ro * (Rmin * x / Ro) ** -1.5)
        #c2 = 15.2 - 0.5 * np.log(0.1 * ne)
        #Nustar = c1 / te ** 2 * (c2 + np.log(te))

        # NN training set values: 
        if self.zerooutpinch == 1: #zero out outward pinch
            veGB[veGB > 0] = 0;

        if self.whenzero == 1:
            filter = np.where(chiiGB < 0);
        elif self.whenzero == 2:
            filter = np.where(chiiGB < 0 | chieGB < 0);
        elif self.whenzero == 3:
            filter = np.where(chiiGB < 0 | chieGB < 0 | deGB< 0);
        elif self.whenzero == 4:
            filter = np.where(chieGB < 0);
        else:
            filter = np.full(chieGB.shape[1], False)
        chiiGB[0, filter] = 0
        chieGB[0, filter] = 0
        deGB[0, filter] = 0
        veGB[0, filter] = 0


        if output_SI:
            # Convert to SI output
            chii = chiiGB * teSI ** (3/2) * chifac
            chie = chieGB * teSI ** (3/2) * chifac
            de = deGB * teSI ** (3/2) * chifac
            ve = veGB * teSI ** (3/2) * chifac / R0
            return np.squeeze(chie), np.squeeze(chii), np.squeeze(de), np.squeeze(ve)
        else:
            return np.squeeze(chieGB), np.squeeze(chiiGB), np.squeeze(deGB), np.squeeze(veGB)

if __name__ == '__main__':
    scann = 24
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(4,10, scann))
    input['Ate'] = input['Ati']
    input['q'] = 2.5
    input['smag'] = 1.2
    input['Ti'] = 5e3
    input['Te'] = 5e3
    R0 = 3
    a0 = 1
    B0 = 5
    Amain = 2
    nn = QLKNN4Dkin()
    fluxes = nn.get_fluxes(
        input['Te'].values,
        input['Ti'].values,
        input['Ate'].values,
        input['Ati'].values,
        input['q'].values,
        input['smag'].values,
        R0, a0, B0, Amain,
        output_SI=True)
    output = pd.DataFrame({'qe_SI': fluxes[0],
                           'qi_SI': fluxes[1],
                           'de_SI': fluxes[2],
                           've_SI': fluxes[3]})
    df = input.join(output)
    print(df)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim(4, 10)
    ax2.set_xlim(4, 10)
    ax1.set_ylim(-5, 20)
    ax2.set_ylim(-1.2, 3)
    df.plot(x='Ati', y=['qe_SI', 'qi_SI'], ax=ax1)
    df.plot(x='Ati', y=['de_SI', 've_SI'], ax=ax2)
    plt.show()
