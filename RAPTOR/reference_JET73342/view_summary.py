import pandas as pd
import numpy as np
from scipy.io import loadmat
from IPython import embed

summary_mat = loadmat('./summary.mat')
fields = summary_mat['summary'].dtype.fields.keys()
rho = summary_mat['summary']['rho'][0,0]
rhovar = pd.DataFrame(rho, columns=['rho'])
const = pd.Series()
for field in fields:
    val = summary_mat['summary'][field][0,0]
    if field == 'tgrid':
        tgrid = np.squeeze(val)
    elif len(val) == 1:
        const[field] = val[0,0]
    elif len(val) == len(rho):
        rhovar[field] = val
    else:
        raise Exception('Could not interpret {!s}'.format(field))
print('Constants:')
print(const)
print('Profiles:')
print(rhovar)
