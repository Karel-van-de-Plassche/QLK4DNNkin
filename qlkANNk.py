from IPython import embed
import numpy as np
import json
import scipy as sc
import scipy.io as sio
import os
from itertools import chain
from collections import OrderedDict
from warnings import warn

import pandas as pd

def sigmo(x):
    #nonlinear neuron function. 
    return 2./(1 + np.exp(-2 * x)) - 1

def NNoutput(Ati, qx, Ti_Te, smag, net):
    max_qlk = net['max_qlk']
    min_qlk = net['min_qlk']
    prepros = net['prepros']
    #initialize input matrix in order expected by the trained network. 
    invec = np.vstack([qx, smag, Ti_Te, Ati])

    #find size of input vector (if Ati, q, Ti_Te, shear are not same length, the function will crash)
    npoints = len(Ati)

    #Normalize input matrix to match NN input normalization
    scalefac = QuaLiKiz4DNN.scalefac
    dispfac  = QuaLiKiz4DNN.dispfac

    #normalize input vector to match input normalization in NN
    unorm = np.multiply(np.add(invec, -dispfac), 1./scalefac)

    #calculate regression neural network piece by piece, for all radii
    g = np.add(np.dot(net['IW'], unorm), net['b1'])
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

class QuaLiKiz4DNN():
    validity_range = {'Zeffx':  (1, 1),
                      'An':     (2, 2),
                      'Ate':    (6, 6),
                      'Ati':    (2, 12),
                      'Nustar': (0.0084058, 0.0084058),
                      'Ti_Te':  (.3, 3),
                      'qx':     (1, 5),
                      'smag':   (.1, 3),
                      'x':      (.5, .5)}
    scalefac = np.array([2 , 1.45 , 1.35 , 5], ndmin=2).T
    dispfac  = np.array([3 , 1.55 , 1.65 , 7], ndmin=2).T
    feature_names = ['qx', 'smag', 'Ti_Te', 'Ati']
    _feature_names = pd.Series(feature_names)
    target_names = ['efeITG_GB', 'efiITG_GB', 'pfeITG_GB']
    _target_names = pd.Series(target_names)

    def __init__(self, whenzero=1, zerooutpinch=0):
        """
        whenzero: since NN was only trained on positive fluxes, they will extrapolate
        to negative flux (which is unphysical for heat fluxes and particle diffusion).
        thus, we zero out all fluxes when:
            - whenzero=1: ion heat fluxes are negative 
            - whenzero=2: ion or electron heat fluxes are negative
            - whenzero=3: ion or electron heat or electron particle diffusion is negative 

        zerooutpinch: if 1, then zeros out positive pinches (not physical for ITG)
        """
        self.whenzero = whenzero
        self.zerooutpinch = zerooutpinch
        netnames = ['netief', 'neteef', 'netdfe', 'netvte', 'netvce']
        self.nets = {}
        root = os.path.dirname(os.path.realpath(__file__))
        for name in netnames:
            with open(os.path.join(root, '4D-NN-' + name + '.json')) as file:
                json_dict = json.load(file)
            self.nets[name] = {key: np.array(val) for key, val in json_dict.items()}

    def get_fluxes(self, Ati=[], qx=[], smag=[], Ti_Te=[], **kwargs):
        for name in kwargs:
            if (np.any(np.less(kwargs[name], self.validity_range[name][0])) or
                np.any(np.greater(kwargs[name] > self.validity_range[name][1]))):
                warn('Using NN out of bounds!')
        for value, name in zip([Ati, qx, Ti_Te, smag], ['Ati', 'qx', 'smag', 'Ti_Te']):
            if (np.any(np.less(value, self.validity_range[name][0])) or
                np.any(np.greater(value, self.validity_range[name][1]))):
                warn('Using NN out of bounds!')

        chii = NNoutput(Ati, qx, Ti_Te, smag, self.nets['netief'])
        chie = NNoutput(Ati, qx, Ti_Te, smag, self.nets['neteef'])
        D    = NNoutput(Ati, qx, Ti_Te, smag, self.nets['netdfe'])
        Vt   = NNoutput(Ati, qx, Ti_Te, smag, self.nets['netvte'])
        Vc   = NNoutput(Ati, qx, Ti_Te, smag, self.nets['netvce'])

        te = 8.
        ne = 5.
        ni = 5.
        Ate = 6.
        Ane = 2.

        Amain = 2.
        b0 = 3.
        r0 = 3.
        Ro = r0
        Rmin = 1.
        x = .5
        a = 1.
        zeff = 1.
        c1 = (6.9224e-5 * zeff * ne * qx * Ro * (Rmin * x / Ro) ** -1.5)
        c2 = 15.2 - 0.5 * np.log(0.1 * ne)
        Nustar = c1 / te ** 2 * (c2 + np.log(te))

        qel=1.6e-19;
        chifac = (te*1e3*qel)**1.5*np.sqrt(Amain*1.67e-27)/(qel**2*b0**2.*a)
        # This kinetic electron 4D NN was (mistakingly) fit to unnormalized D and V. We must normalize back to GB using the
        # Te, B0, and a values used when constructing the NN database, and then unnormalize again using
        # the simulation values

        # NN training set values: 
        #Te = 8 keV , a = 1 m, B0 = 3 T, Amain = 2, n = 5e19 [m^-3], R/Ln=2, R/LTe=6, x=0.5, Zeff=1
        chifacNN = (8e3*qel)**1.5*np.sqrt(2*1.67e-27)/(qel**2.*3**2.*1);

        chii=(chifac*chii); #convert to SI
        chie=(chifac*chie); #convert to SI
        D=D * chifac/chifacNN; #convert from SI to GB with NN units, then to SI with simulation units
        V=(Vt+ Vc) *chifac/chifacNN * 3./r0
        if self.zerooutpinch == 1: #zero out outward pinch (not expected for ITG)
            V[V>0]=0;
        if self.whenzero == 1:
            filter = np.where(chii<0);
        elif self.whenzero == 2:
            filter = np.where(chii<0 | chie<0);
        elif self.whenzero == 3:
            filter = np.where(chii<0 | chie<0 | D<0);
        chii[filter]=0;
        chie[filter]=0;
        D[filter]=0;
        V[filter]=0;

        dndr = -Ane/r0*(ne*1e19);
        dtidr = -Ati/r0*(te*Ti_Te*1e3*qel);
        dtedr = -Ate/r0*(te*1e3*qel);

        mistake_factor = 1.3
        qe_GB = mistake_factor * -chie/chifac*dtedr/(te*1e3*qel/r0);
        qi_GB = mistake_factor * -chii/chifac*dtidr/(te*Ti_Te*1e3*qel/r0);
        pfe_GB = mistake_factor * (-D*dndr+V*ne*1e19)/chifac/(ne*1e19/r0);

        return np.squeeze(qe_GB), np.squeeze(qi_GB), np.squeeze(pfe_GB)

    def get_output(self, input, clip_low=True, clip_high=True, low_bound=None, high_bound=None, safe=True, output_pandas=True, output_vars=None):
        if output_vars is None:
            output_vars = self._target_names
        if safe is True:
            input = input[self._feature_names]
            input = {col: input[col].as_matrix() for col in input}
        else:
            input = {name: col for name, col in zip(self._feature_names, input.T)}
        output = np.vstack(self.get_fluxes(**input))
        output = clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound)
        out_dict = OrderedDict()
        for name, out in zip(self.target_names, output):
            if name in output_vars:
                out_dict[name] = out

        result = pd.DataFrame(out_dict)
        if output_pandas is False:
            result = result.as_matrix()
        return result

def clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound):
    if clip_low:
        for ii, bound in enumerate(low_bound):
            output[:, ii][output[:, ii] < bound] = bound

    if clip_high:
        for ii, bound in enumerate(high_bound):
            output[:, ii][output[:, ii] > bound] = bound

    return output

if __name__ == '__main__':
    scann = 24
    Ati = np.array(np.linspace(2,13, scann))
    qx = np.full_like(Ati, 2.)
    smag = np.full_like(Ati, 1.)
    Ti_Te = np.full_like(Ati, 1.)
    nn = QuaLiKiz4DNN()
    fluxes = nn.get_fluxes(Ati, qx, smag, Ti_Te, Ate=[5])

    scann = 100
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    input['Zeffx']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 4.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['qx'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['x']  = np.full_like(input['Ati'], 0.449951)
    nn.get_output(input)

    embed()

