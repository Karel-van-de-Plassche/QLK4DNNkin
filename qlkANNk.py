from IPython import embed
import numpy as np
import json
import scipy as sc
import scipy.io as sio
def sigmo(x):
    #nonlinear neuron function. 
    return 2./(1 + np.exp(-2 * x)) - 1

def NNoutput(Ati, q, tite, smag, net):
    max_qlk = net['max_qlk']
    min_qlk = net['min_qlk']
    prepros = net['prepros']
    #initialize input matrix in order expected by the trained network. 
    invec = np.vstack([q, smag, tite, Ati])

    #find size of input vector (if Ati, q, tite, shear are not same length, the function will crash)
    npoints = len(Ati)

    #Normalize input matrix to match NN input normalization
    scalefac = np.array([2 , 1.45 , 1.35 , 5], ndmin=2).T
    dispfac  = np.array([3 , 1.55 , 1.65 , 7], ndmin=2).T

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
    validity_range = {'Ane':    (2, 2),
                      'Zeffx':  (1, 1),
                      'An':     (2, 2),
                      'Ate':    (6, 6),
                      'Ati':    (2, 12),
                      'Nustar': (0.0084058, 0.0084058),
                      'Ti_Te':  (.3, 3),
                      'qx':     (1, 5),
                      'smag':   (.1, 3),
                      'x':      (.5, .5)}

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
        for name in netnames:
            with open('4D-NN-' + name + '.json') as file:
                json_dict = json.load(file)
            self.nets[name] = {key: np.array(val) for key, val in json_dict.items()}

    def get_fluxes(self, Ati, q, smag, tite):
        chii = NNoutput(Ati, q, tite, smag, self.nets['netief'])
        chie = NNoutput(Ati, q, tite, smag, self.nets['neteef'])
        D    = NNoutput(Ati, q, tite, smag, self.nets['netdfe'])
        Vt   = NNoutput(Ati, q, tite, smag, self.nets['netvte'])
        Vc   = NNoutput(Ati, q, tite, smag, self.nets['netvce'])

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
        c1 = (6.9224e-5 * zeff * ne * q * Ro * (Rmin * x / Ro) ** -1.5)
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
        dtidr = -Ati/r0*(te*tite*1e3*qel);
        dtedr = -Ate/r0*(te*1e3*qel);
        
        qe_GB = -chie/chifac*dtedr/(te*1e3*qel/r0);
        qi_GB = -chii/chifac*dtidr/(te*tite*1e3*qel/r0);
        pfe_GB = (-D*dndr+V*ne*1e19)/chifac/(ne*1e19/r0);

        return np.squeeze(qe_GB), np.squeeze(qi_GB), np.squeeze(pfe_GB)

if __name__ == '__main__':
    scann = 24
    Ati = np.array(np.linspace(2,13, scann))
    q = np.full_like(Ati, 2.)
    smag = np.full_like(Ati, 1.)
    tite = np.full_like(Ati, 1.)
    nn = QuaLiKiz4DNN()
    nn.get_fluxes(Ati, q, smag, tite)
    
    embed()

